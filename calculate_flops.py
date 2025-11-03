import torch
from thop import profile
from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    # 实例化模型
    # 根据您的实际模型配置修改参数
    # 例如，如果您的backbone是xception，则改为DeepLab(num_classes=8, backbone="xception")
    model = DeepLab(num_classes=8, backbone="mobilenetv2") 
    
    # 创建一个虚拟输入，用于计算FLOPs和参数量
    # 输入尺寸应与模型期望的输入尺寸一致，例如 [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 3, 512, 512) 
    
    # 计算FLOPs和参数量
    total_ops, total_params = profile(model, (input_tensor,), verbose=False)
    
    print("模型参数量 (M): {:.2f}".format(total_params / 1e6))
    print("模型FLOPs (G): {:.2f}".format(total_ops / 1e9))