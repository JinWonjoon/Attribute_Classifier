import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import vgg16, resnet18, resnet50, wide_resnet50_2

import sys
import math

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = vgg16(pretrained=True)
#        self.vgg.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.vgg(x)
        return torch.sigmoid(x)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        # self.resnet_feature = nn.Sequential(*list(resnet18(pretrained=True).modules())[:-1]) # strips off last linear layer
        self.resnet_feature = resnet18(pretrained=True)
        self.resnet_feature.fc = Identity()
        self.fc_landmark = nn.Linear(512, 24)
        self.fc_attribute = nn.Linear(512, 3)

    def forward(self, x):
        x = self.resnet_feature(x)
        landmark = self.fc_landmark(x)
        attribute = self.fc_attribute(x)
        return F.sigmoid(attribute), landmark

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet_feature = resnet50(pretrained=True)
        self.resnet_feature.fc = Identity()
        self.fc_landmark = nn.Linear(2048, 24)
        self.fc_attribute = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.resnet_feature(x)
        landmark = self.fc_landmark(x)
        attribute = self.fc_attribute(x)
        return F.sigmoid(attribute), landmark

class WideResNet50_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.wresnet = wide_resnet50_2(pretrained=True)
        self.wresnet.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.wresnet(x)
        return F.sigmoid(x)


if __name__ == '__main__':
    net = ResNet18()