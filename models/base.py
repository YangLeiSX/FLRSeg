#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: base.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: base block to build the network
#
#================================================================

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.utils import init_weights

class PSLayer(nn.Module):
    def __init__(self, 
            low_ch: int, 
            high_ch: int
        ):
        super(PSLayer, self).__init__()
        self.ps = nn.PixelShuffle(2)
        self.conv_ps = nn.Sequential(
            nn.Conv2d(high_ch, low_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(low_ch * 4),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(low_ch, low_ch, kernel_size=1),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True)
        )
        self.apply(init_weights)

    def forward(self, x_low, x_high):
        x_ps = self.ps(self.conv_ps(x_high))
        out = self.conv(x_ps+x_low)

        return out

class Decoder(nn.Module):
    def __init__(self,
            channels: List[int] = [64, 128, 256, 512],
            scales: List[int] = [1, 2, 4, 8]
        ):
        super().__init__()
        self.scales = scales
        self.up1 = PSLayer(channels[0], channels[1])
        self.up2 = PSLayer(channels[1], channels[2])
        self.up3 = PSLayer(channels[2], channels[3])
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(channels[:3]), channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True)
        )
        self.apply(init_weights)

    def forward(self, f1, f2, f3, f4):
        up3 = self.up3(f3, f4)
        up2 = self.up2(f2, up3)
        up1 = self.up1(f1, up2)

        return up1

class SEBlock(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int
        ):
        super(SEBlock, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.extract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        weight = self.se(x)
        x = x * weight

        res = self.extract(x)
        return res

class SegOut(nn.Module):
    def __init__(self, 
            in_ch: int, 
            n_classes: int,
            scale: int = 4
        ):
        super(SegOut, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)
        self.apply(init_weights)

    def forward(self, x):
        res = self.conv(x)
        res = F.interpolate(res, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return res

class FCNOut(nn.Module):
    def __init__(self, 
            in_ch: List[int],
            n_classes: int,
            scales: List[int]
        ):
        super(FCNOut, self).__init__()
        self.scales = scales
        self.conv = nn.Conv2d(sum(in_ch), n_classes, kernel_size=1)
        self.apply(init_weights)

    def forward(self, feats:List[Tensor]):
        scaled_feats = [F.interpolate(
            feats[i], scale_factor=scale, mode='bilinear', align_corners=True)
            for i, scale in enumerate(self.scales)]
        return self.conv(torch.cat(scaled_feats, dim=1))

