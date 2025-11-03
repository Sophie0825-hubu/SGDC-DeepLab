import torch
import torch.nn as nn
from torchvision import models

def vgg16(pretrained=False, **kwargs):
    """
    VGG16 model implementation.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
    """
    model = models.vgg16(pretrained=pretrained)
    return model

class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG, self).__init__()
        vgg16_model = vgg16(pretrained=pretrained)
        features = list(vgg16_model.features.children())
        
        # Extract features up to the last maxpool layer
        self.features = nn.Sequential(*features[:-1])
        
        # Add a 1x1 convolution to reduce the channel dimension
        self.reduce_conv = nn.Conv2d(512, 256, kernel_size=1)
        
        # Extract features up to the 4th maxpool layer
        self.low_level_features = nn.Sequential(*features[:23])

    def forward(self, x):
        low_level_features = self.low_level_features(x)
        x = self.features(x)
        x = self.reduce_conv(x)
        return low_level_features, x