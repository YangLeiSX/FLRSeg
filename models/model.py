#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: model.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: 
#
#================================================================

from typing import List
from collections import namedtuple

import torch
import torch.nn as nn

from models.base import Decoder, SEBlock, SegOut, FCNOut
from models.resnet import ResNet34_Encoder

from models.fuzzy import FuzzyLearning
from models.graph import FeatGR

ModelOut = namedtuple('ModelOut', ['pred', 'aux', 'bound'])

class _SegNet(nn.Module):
    def __init__(self, 
            n_classes: int, ch_in: int = 3,
            num_fuzzy: List[int] = None, type_fz: nn.Module = None,
            num_node: List[int] = None, type_gr: nn.Module = None,
            channels: List[int] = [64, 128, 256, 512],
            scales: List[int] = [4, 8, 16, 32],
            device: str = 'cuda'
        ):
        super(_SegNet, self).__init__()
        self.n_classes = n_classes
        self.num_fuzzy = num_fuzzy
        self.num_node = num_node
        self.device = device

        self.encoder = ResNet34_Encoder()
        self.decoder = Decoder(channels, [s/scales[0] for s in scales])

        se_scale = 1
        if type_fz is not None:
            self.fz_modules = nn.ModuleList([
                eval(f"{type_fz}{channels[i], num_fuzzy[i], channels[i]//4}")
                for i in range(len(num_fuzzy))
            ])
            se_scale = se_scale + 1
        else:
            self.fz_modules = None

        if type_gr is not None:
            self.gr_modules = nn.ModuleList([
                eval(f"{type_gr}{channels[i], num_node[i], channels[i]//4}")
                for i in range(len(num_fuzzy))
            ])
            se_scale = se_scale + 1
        else:
            self.gr_modules = None

        self.se_blocks = nn.ModuleList([SEBlock(ch*se_scale, ch) for ch in channels[:3]])

        self.aux = FCNOut(channels, n_classes, scales)
        self.bound = SegOut(channels[0], 1, scale=scales[0])
        self.seg = SegOut(channels[0], n_classes, scale=scales[0])

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        feats = list(feats)
        aux_out = self.aux(feats)

        for i in range(3):
            feat_se = [feats[i]]
            if self.fz_modules is not None:
                feat_se.append(self.fz_modules[i](feats[i]))

            if self.gr_modules is not None:
                feat_se.append(self.gr_modules[i](feats[i]))

            feats[i] = self.se_blocks[i](torch.cat(feat_se, dim=1))

        res = self.decoder(*feats)

        pred_out = self.seg(res)
        bound_out = torch.sigmoid(self.bound(res))

        return ModelOut(
            pred=pred_out,
            aux=aux_out,
            bound=bound_out
        )

def SegNet(**kwargs):
    return _SegNet(type_fz=None, type_gr=None, **kwargs)

def FLRSegNet(**kwargs):
    return _SegNet(type_fz='FuzzyLearning', type_gr='FeatGR', **kwargs)

