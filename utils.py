#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: utils.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: 
#
#================================================================

import torch

def get_parameter_number(net):
    """
    get_parameter_number 计算模型中的参数数量

    net(nn.Module):      需要计算参数数量的模型
    """
    total_num = sum(p.numel() for p in net.parameters())
    return total_num

def flip_tensor(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]

def resize_tensor(inputs, target_size):
    inputs = torch.nn.functional.interpolate(
        inputs, size=target_size, mode='bilinear',
        align_corners=False)
    return inputs
