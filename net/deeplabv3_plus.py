import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50
from nets.vgg import VGG
import math

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]
        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # 原代码
        # low_level_features = self.features[:4](x)
        # x = self.features[4:](low_level_features)
        feature1 = self.features[:2](x)
        print(feature1.shape)
        #assert False

        # 修改后代码
        feature1 = self.features[:2](x) # [1, 16, 256, 256]   2
        feature2 = self.features[2:8](feature1) # [1, 64, 64, 64]   8
        x = self.features[8:](feature2) # [1, 320, 64, 64]   17

        # return low_level_features, x 
        return feature1, feature2, x 

class GaussianDilatedConv(nn.Module):
    """高斯-空洞混合卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, sigma=1.0, gate_fn=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = dilation * (kernel_size - 1) // 2
        
        # 主卷积层（空洞卷积）
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        
        # 高斯卷积层
        self.gaussian_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=False
        )
        
        # 初始化高斯核权重
        self._init_gaussian_kernel(sigma)
        
        # 特征融合
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
        )
        
        # 残差连接：使用空洞卷积的特征
        self.residual = nn.Conv2d(in_channels, out_channels, 1, bias=False) if gate_fn is None else gate_fn(in_channels, out_channels, 1, bias=False)
        
        # 门控机制
        self.gate = nn.Sigmoid() if gate_fn is None else gate_fn()

    def _init_gaussian_kernel(self, sigma):
        """初始化高斯核权重"""
        kernel_size = self.gaussian_conv.kernel_size[0]
        device = next(self.parameters()).device

        x_cord = torch.arange(kernel_size, device=device)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # 计算2D高斯分布
        kernel = (1. / (2. * math.pi * variance)) * \
                 torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        kernel = kernel / torch.sum(kernel)  # 归一化
        
        # 应用高斯核到所有输入通道
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(
            self.gaussian_conv.out_channels, self.gaussian_conv.in_channels // self.gaussian_conv.groups, 1, 1
        )
        
        # 固定高斯权重，不参与训练
        self.gaussian_conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        # 保存原始输入尺寸
        H, W = x.size(2), x.size(3)
        
        # 计算正确的输出尺寸
        output_size = (
            H + 2 * self.dilated_conv.padding[0] - self.dilation * (self.kernel_size - 1) - 1
        ) // self.dilated_conv.stride[0] + 1
        
        # 空洞卷积路径
        dilated_path = self.dilated_conv(x)
        
        # 高斯路径 - 使用自适应池化确保尺寸匹配
        gaussian_path = self.gaussian_conv(x)
        if gaussian_path.shape[2:] != dilated_path.shape[2:]:
            gaussian_path = F.adaptive_avg_pool2d(
                gaussian_path, 
                output_size=(dilated_path.size(2), dilated_path.size(3))
            )
        
        # 特征融合
        fused = dilated_path + gaussian_path
        fused = self.fuse(fused)
        
        # 残差连接：使用空洞卷积的特征
        residual = self.residual(x)
        if residual.shape[2:] != fused.shape[2:]:
            residual = F.adaptive_avg_pool2d(
                residual, 
                output_size=(fused.size(2), fused.size(3))
            )
        
        # 应用门控机制
        gate = self.gate(fused) if hasattr(self, 'gate') else torch.ones_like(fused)
        output = gate * fused + (1 - gate) * residual
        
        # 恢复原始尺寸
        if output.size(2) != H or output.size(3) != W:
            output = F.interpolate(
                output, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=True
            )

        return output

# 修改后的增强版ASPP
class EnhancedASPP(nn.Module):
    """增强版ASPP，使用高斯-空洞混合卷积"""
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        # 使用高斯-空洞混合卷积替换原ASPP分支
        self.branch2 = nn.Sequential(
            GaussianDilatedConv(dim_in, dim_out, 3, dilation=6*rate, sigma=1.2),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            GaussianDilatedConv(dim_in, dim_out, 3, dilation=12*rate, sigma=1.5),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            GaussianDilatedConv(dim_in, dim_out, 3, dilation=18*rate, sigma=1.8),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 处理各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # 全局特征分支
        global_feature = self.branch5(x)
        global_feature = F.interpolate(global_feature, (h, w), mode='bilinear', align_corners=True)
        
        # 特征拼接
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="mobilenetv2":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif backbone=="resnet50":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = resnet50(pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="vgg":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [64, 64, 512]
            #   主干部分    [32, 32, 256]
            #----------------------------------#
            self.backbone = VGG(pretrained=pretrained)
            in_channels = 256
            low_level_channels = 512

        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        #self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #-----------------------------------------#
        #   使用增强版ASPP特征提取模块
        #   结合高斯-空洞混合卷积
        #-----------------------------------------#
        self.aspp = EnhancedASPP(
            dim_in=in_channels, 
            dim_out=256, 
            rate=16//downsample_factor
        )
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		

        self.cat_conv = nn.Sequential(
            # nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.Conv2d(416, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)


        self.ghostFusionModule = GhostFusionModule()

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        # low_level_features, x = self.backbone(x)

        # [1, 16, 256, 256] [1, 64, 64, 64] [1, 320, 64, 64]
        feature1, feature2, x = self.backbone(x)

        ghostFusionFeatures = self.ghostFusionModule(feature1, feature2, x)

        aspp_x = self.aspp(x)    
        
        # [1, 416, 64, 64]
        # low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        aspp_x = F.interpolate(aspp_x, size=(ghostFusionFeatures.size(2), ghostFusionFeatures.size(3)), mode='bilinear', align_corners=True)
        print(aspp_x.shape)  # 优化后深层特征尺寸
        print(ghostFusionFeatures.shape)  # LFFM后特征尺寸
        x = self.cat_conv(torch.cat((aspp_x, ghostFusionFeatures), dim=1))
        
        #y = torch.cat((aspp_x, ghostFusionFeatures), dim=1)
        #print(y.shape)  # 拼接后特征尺寸
        
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        #print(x.shape)  # 最终输出
        return x

class GhostModule(nn.Module):

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostFusionModule(nn.Module):
    def __init__(self):
        super(GhostFusionModule, self).__init__()

        # A路径: 16x256x256 -> 32x128x128 -> 64x64x64
        self.path_a = nn.Sequential(
            GhostModule(16, 32, kernel_size=3, stride=2),  # 下采样到128x128
            GhostModule(32, 64, kernel_size=3, stride=2)  # 下采样到64x64
        ) # 64

        # B路径: 32x64x64 -> 64x64x64 (点卷积)
        self.path_b = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        # 融合路径: 128x64x64 -> 160x32x32
        self.fusion = GhostModule(128, 160, kernel_size=3, stride=1)

        # 最终点卷积: 160x32x32 -> 160x32x32
        self.final_conv = nn.Conv2d(480, 160, kernel_size=1, stride=1)

        self.downsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 64
        )

    def forward(self, x_a, x_b, x_external):
        # 处理A路径
        a_out = self.path_a(x_a)

        # 处理B路径
        b_out = self.path_b(x_b)

        # 拼接A和B的输出
        fused = torch.cat([a_out, b_out], dim=1)

        # 融合处理
        fused_out = self.fusion(fused)

        # 与外部特征图进行点卷积融合
        result = self.final_conv(torch.cat([fused_out, x_external], dim=1))  # 使用残差连接
        output = self.downsample(result)
        
        return output