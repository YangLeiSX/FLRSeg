#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: graph.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: fuzzy garph reasoning module
#
#================================================================

import torch
import torch.nn as nn

class GCN_Conv(nn.Module):
    def __init__(self, node, channel):
        super(GCN_Conv, self).__init__()

        self.conv1d_1 = nn.Conv1d(node, node, 1)
        self.conv1d_2 = nn.Conv1d(channel, channel, 1)

    def forward(self, x):
        # K * N
        h = self.conv1d_1(x) - x
        h = h.permute(0, 2, 1)
        # N * K
        return self.conv1d_2(h).permute(0, 2, 1)


class FeatGR(nn.Module):
    def __init__(self,
        in_channel: int,
        n_node: int,
        node_feature: int,
        gcn_times: int = 2
    ) -> None:
        super().__init__()

        self.in_channel = in_channel
        self.n_node = n_node 
        self.gcn_times = gcn_times
        self.node_feature = node_feature

        self.reduce = nn.Conv2d(in_channel, node_feature, 1)
        self.extend = nn.Conv2d(node_feature, in_channel, 1)

        self.theta = nn.Conv2d(in_channel, n_node, 1)
        self.gcns = nn.ModuleList(
            [GCN_Conv(n_node, node_feature)
                for _ in range(gcn_times)]
        )

        self.mu = nn.Parameter(torch.randn(1, n_node, node_feature), requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(1, 1, n_node), requires_grad=True)

    def forward(self, x):
        bs, c, h, w = x.shape
        K, D, N = self.n_node, self.node_feature, h*w

        # reduce dim to [B, N, D]
        # [B, C, H, W] -> [B, D, H, W] -> [B, D, N]
        x_reduced = self.reduce(x).reshape(bs, D, N)

        # Generate Membership
        # [B, N, D] & [B, K, D] -> [B, N, K]
        # member = torch.cdist(x_reduced.permute(0, 2, 1), self.mu.repeat(bs, 1, 1))
        # member = torch.exp(-( member / self.sigma.repeat(bs, N, 1))**2)

        member = torch.cdist(x_reduced.permute(0, 2, 1), self.mu.expand(bs, -1, -1))
        member = torch.exp(-( member / self.sigma.expand(bs, N, -1))**2)

        # project 
        # [B, D, N] @ [B, N, K] @  -> [B, D, K] -> [B, K, D]
        feat = torch.bmm(
            x_reduced, 
            # torch.softmax(member, dim=1), 
            nn.functional.normalize(member, dim=1)
        )

        # gcn [B, K, D]
        gcn_feat = feat.permute(0, 2, 1)
        for i in range(self.gcn_times):
            gcn_feat = torch.sigmoid(self.gcns[i](gcn_feat))
        feat = feat.permute(0, 2, 1) + gcn_feat

        # re-project
        # [B, N, K] @ [B, K, D] -> [B, N, D]
        x_back = torch.bmm(
            # torch.softmax(member, dim=2),
            nn.functional.normalize(member, dim=2),
            feat
        )

        # [B, N, D] -> [B, D, N] -> [B, D, H, W]
        x_back = x_back.permute(0, 2, 1).reshape(bs, D, h, w)

        # extent dim
        # [B, D, H, W] -> [B, C, H, W]
        res = self.extend(x_back)

        return torch.relu(x + res)

