#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: resnet.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: resnet encoder
#
#================================================================

import torch
from torch import nn
from torchvision.models import resnet34, resnet18

class ResNet34_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        basenet = resnet34(pretrained=True)
        self.stage0 = nn.Sequential(
            basenet.conv1,
            basenet.bn1,
            basenet.relu,
            basenet.maxpool
        )
        self.stage1 = basenet.layer1
        self.stage2 = basenet.layer2
        self.stage3 = basenet.layer3
        self.stage4 = basenet.layer4

    def forward(self, x):
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return f1, f2, f3, f4

class ResNet18_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        basenet = resnet18(pretrained=True)
        self.stage0 = nn.Sequential(
            basenet.conv1,
            basenet.bn1,
            basenet.relu,
            basenet.maxpool
        )
        self.stage1 = basenet.layer1
        self.stage2 = basenet.layer2
        self.stage3 = basenet.layer3
        self.stage4 = basenet.layer4

    def forward(self, x):
        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return f1, f2, f3, f4
