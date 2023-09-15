#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: fuzzy.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: fuzzy learning module
#
#================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class FuzzyLearning(nn.Module):
    def __init__(self, in_channel, fuzzynum, fuzzychannel, T=20) -> None:
        super(FuzzyLearning, self).__init__()
        self.n = fuzzynum
        self.T = T

        self.conv1 = nn.Conv2d(in_channel, fuzzychannel, 3, padding=1)
        self.conv2 = nn.Conv2d(3 * fuzzychannel, in_channel, 3, padding=1)

        self.mu = nn.Parameter(torch.randn((fuzzychannel, self.n)))
        self.sigma = nn.Parameter(torch.randn((fuzzychannel, self.n)))

    def forward(self, x):
        x = self.conv1(x)
        feat = x.permute((0, 2, 3, 1))

        member = feat.unsqueeze(-1).expand(-1, -1, -1, -1, self.n)
        member = torch.exp(-((member - self.mu)/self.sigma) ** 2)

        sample = torch.randn_like(member) * self.sigma + self.mu

        member_and = torch.sum(
            sample * torch.softmax((1-member) * self.T, dim=4),
            dim=4).permute((0, 3, 1, 2))

        member_or = torch.sum(
            sample * torch.softmax(member * self.T, dim=4),
            dim=4).permute((0, 3, 1, 2))

        feat = torch.cat([x, member_and, member_or], dim=1)
        feat = self.conv2(feat)
        return feat
