#! /usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#
#   file name: predict.py
#   author: tripletone
#   email: yangleisxn@gmail.com
#   created date: 2023/09/15
#   description: 
#
#================================================================

import os
import argparse
import itertools
from functools import partial

import cv2
import torch
import numpy as np
from PIL import Image

from models.model import FLRSegNet
from utils import get_parameter_number, flip_tensor, resize_tensor

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

@torch.no_grad()
def predict_img_ms(net, full_img, device):
    net.eval()
    src_h, src_w = full_img.size[1], full_img.size[0]
    img = torch.from_numpy(preprocess(full_img, align=80))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    scales = [0.4, 0.8, 1.2, 1.6]
    vflips = [0]
    hflips = [0, 1]
    input_size = img.shape[2], img.shape[3]
    pred_buf = torch.zeros((1, 5, img.shape[2], img.shape[3]), device=device)
    
    for vflip, hflip, scale in itertools.product(vflips, hflips, scales):
        inputs = img
        infer_size = [round(sz * scale) for sz in input_size]

        inputs = flip_tensor(inputs, 2) if vflip == 1 else inputs
        inputs = flip_tensor(inputs, 3) if hflip == 1 else inputs
        inputs = resize_tensor(inputs, infer_size) if scale != 1.0 else inputs

        output = net(inputs)
        pred = output.pred

        _pred = resize_tensor(pred, input_size) if scale != 1.0 else pred
        _pred = flip_tensor(_pred, 2) if vflip == 1 else _pred
        _pred = flip_tensor(_pred, 3) if hflip == 1 else _pred

        pred_buf = pred_buf + _pred

    pred = pred_buf / len(scales) / len(vflips) / len(hflips)

    # 1, 5, H, W
    h, w = pred.shape[2], pred.shape[3]
    full_mask = pred[0, :,
                    (h-src_h)//2: h//2+src_h-src_h//2,
                    (w-src_w)//2: w//2+src_w-src_w//2].cpu()
    # 5, h, w
    res = torch.argmax(full_mask, dim=0).numpy()

    return res

@torch.no_grad()
def predict_img(net, full_img, device):
    net.eval()
    src_h, src_w = full_img.size[1], full_img.size[0]
    img = torch.from_numpy(preprocess(full_img))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    output = net(img)
    pred = output.pred

    # 1, 5, H, W
    h, w = pred.shape[2], pred.shape[3]
    full_mask = pred[0, :, 
                    (h-src_h)//2: h//2+src_h-src_h//2,
                    (w-src_w)//2: w//2+src_w-src_w//2].cpu()
    # 5, h, w
    res = full_mask.argmax(dim=0).numpy()

    return res

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', type=str, default='params-b5b19c.pth',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', 
                        help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',   
                        help='Filenames of output images')
    parser.add_argument('--type', type=str, default='demo',
                        help='Model type')
    parser.add_argument('--cuda', action='store_true', 
                        help='Use cuda to predict')
    parser.add_argument('--multiscale', action='store_true', 
                        help='Predict in multiple scale')
    return parser.parse_args()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_output_filenames(args):
    def _generate_name(fn, mtype):
        split = os.path.splitext(fn)
        return f'{split[0]}_{mtype}.png'

    __gen_name = partial(_generate_name, mtype=args.type)
    return args.output or list(map(__gen_name, args.input))

def mask_to_image_color(mask):
    image_color = np.zeros_like(mask, np.uint8)
    image_color = np.stack([image_color]*3).transpose((1,2,0))
    for h, w in itertools.product(range(image_color.shape[0]), range(image_color.shape[1])):
        if mask[h, w] == 1:
            image_color[h, w] = [128, 0, 0]
        elif mask[h, w] == 2:
            image_color[h, w] = [0, 128, 0]
        elif mask[h, w] == 3:
            image_color[h, w] = [128, 128, 0]
        elif mask[h, w] == 4:
            image_color[h, w] = [0, 0, 128]
        else:
            image_color[h, w] = [0, 0, 0]
    return Image.fromarray(image_color)

def preprocess(pil_img, align=64):
    img_ndarray = np.asarray(pil_img)
    h, w, c = img_ndarray.shape
    top, left, bottom, right = 0, 0, 0, 0
    height, width = align*(h//align+1), align*(w//align+1)

    if h > height:
        img_ndarray = img_ndarray[(h-height)//2:(h+height)//2, :, :]
    else:
        top = (height - h) // 2
        bottom = height - h - top

    if w > width:
        img_ndarray = img_ndarray[:, (w-width)//2:(w+width)//2, :]
    else:
        left = (width - w) // 2 
        right = width - w - left

    img_ndarray = cv2.copyMakeBorder(img_ndarray, 
                                     top, bottom, left, right, 
                                     cv2.BORDER_REPLICATE)

    """通道对齐"""
    if img_ndarray.ndim == 2:
        img_ndarray = img_ndarray[..., np.newaxis]
    img_ndarray = img_ndarray.transpose((2, 0, 1))

    """取值范围和标签的对齐"""
    img_ndarray = img_ndarray / 255

    return img_ndarray

if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    # Setup Device
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Using device {device}')

    # Setup Model
    model_args = dict(
        n_classes=5,
        num_fuzzy = [16, 32, 64],
        num_node = [16, 32, 64],
    )
    net = FLRSegNet(**model_args).to(device)
    print(f"#Params: {get_parameter_number(net) * 1e-6} M")

    print(f'Loading model from {args.model}')
    state_dict = torch.load(args.model, map_location=device)['state_dict']
    if '__metric' in state_dict.keys():
        state_dict.pop('__metric')
    net.load_state_dict(state_dict)
    print('Model loaded!')

    for i, filename in enumerate(in_files):
        print(
            f'({i+1}/{len(in_files)}) Predicting image {filename} ...',
            end='\r'
        )

        img = Image.open(filename)

        predict_args = dict(
            net=net,
            full_img=img,
            device=device,
        )
        if args.multiscale:
            mask = predict_img_ms(**predict_args)
        else:
            mask = predict_img(**predict_args)

        torch.cuda.empty_cache()

        out_filename = out_files[i]
        result = mask_to_image_color(mask)
        result = get_concat_h(img, result)
        result.save(out_filename)
        print(f'Mask saved to {out_filename}')
