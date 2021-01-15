import torch
import torch.nn as nn
from augerino import models

def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class smallnet(nn.Module):
    """
    Very small CNN
    """
    def __init__(self, in_channels=3, num_targets=10,k=128,dropout=True):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),  # Inserted to allow fc layer.
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            # Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Flatten(),
            nn.Linear(2*k * 6 * 6,num_targets)
        )
    def forward(self,x):
        return self.net(x)


class AugAveragedConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, aug, ncopies=4, stride=1):
        super().__init__()
        self.aug = aug
        self.conv = nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.ncopies = ncopies

    def forward(self,x):
        out = self.act(self.bn(self.conv(x)))
        return sum([self.aug(out) for _ in range(self.ncopies)]) / self.ncopies


class avgd_smallnet(nn.Module):
    def __init__(self, in_channels=3, num_targets=10,k=128,dropout=True, ncopies=4):
        super().__init__()
        self.num_targets = num_targets
        self.augs = [models.ChannelUniformAug(i) for i in [k, k, 2 * k]]
        self.net = nn.Sequential(
            AugAveragedConvBlock(in_channels,k, self.augs[0], ncopies=ncopies),
            AugAveragedConvBlock(k,k, self.augs[1], ncopies=ncopies),
            AugAveragedConvBlock(k,2*k, self.augs[2], ncopies=ncopies),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),  # Inserted to allow fc layer.
            nn.Dropout2d(.3) if dropout else nn.Sequential(),
            # Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Flatten(),
            nn.Linear(2*k * 6 * 6,num_targets)
        )
    def forward(self,x):
        return self.net(x)


