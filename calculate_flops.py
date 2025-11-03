import torch
from thop import profile
from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":

    model = DeepLab(num_classes=8, backbone="mobilenetv2") 

    input_tensor = torch.randn(1, 3, 512, 512) 
    
    total_ops, total_params = profile(model, (input_tensor,), verbose=False)
    
    print("模型参数量 (M): {:.2f}".format(total_params / 1e6))
    print("模型FLOPs (G): {:.2f}".format(total_ops / 1e9))