"""
InceptionNeXt paper: https://arxiv.org/abs/2303.16900
Original implementation & weights from: https://github.com/sail-sg/inceptionnext
"""

from functools import partial
import copy
import logging
import math
import os
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final


import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, DropPath, to_2tuple, get_padding, SelectAdaptivePool2d
import numpy as np
from time import time
from einops import rearrange, repeat


__all__ = ['InceptFormer']




def multi_retrieval_2D_integral_image(integral_image, window_size):
    
    B, F, H_, W_ = integral_image.shape
    # Step : Generate indices for all pixels (i,j)
    i_indices = torch.arange(H_ - 1, device=integral_image.device)
    j_indices = torch.arange(W_ - 1, device=integral_image.device)
    ii, jj = torch.meshgrid(i_indices, j_indices, indexing='ij')  # (H, W), (H, W)

    # Compute window bounds (clamped to image dimensions)
    r_x = window_size[0] // 2  # Window radius
    r_y = window_size[1] // 2  # Window radius
    x1 = (ii - r_x).clamp(min=0)
    x2 = (ii + r_x).clamp(max=H_-2)
    y1 = (jj - r_y).clamp(min=0)
    y2 = (jj + r_y).clamp(max=W_-2)

    # Convert to integral image coordinates (+1 for padding offset)
    x2_plus_1 = x2 + 1
    y2_plus_1 = y2 + 1

    # # Gather values from integral image using advanced indexing
    # a = integral_image[:, :, x2_plus_1, y2_plus_1]
    # b = integral_image[:, :, x1, y2_plus_1]
    # c = integral_image[:, :, x2_plus_1, y1]
    # d = integral_image[:, :, x1, y1]
    # # Compute the sum using inclusion-exclusion principle
    # output = a - b - c + d

    # Flatten index pairs into single linear index
    def flatten_indices(x_idx, y_idx):
        return x_idx * W_ + y_idx  # shape: (H-1, W-1)

    idx_a = flatten_indices(x2_plus_1, y2_plus_1)
    idx_b = flatten_indices(x1,  y2_plus_1)
    idx_c = flatten_indices(x2_plus_1, y1)
    idx_d = flatten_indices(x1,  y1)

    # Reshape integral image to [B, F, H_*W_] to use gather
    flat_integral = integral_image.view(B, F, -1)  # [B, F, H_*W_]

    # Indices need to be broadcast to [B, F, H-1, W-1] and reshaped
    idx_shape = idx_a.shape
    idx_a = idx_a.view(1, 1, -1).expand(B, F, -1)
    idx_b = idx_b.view(1, 1, -1).expand(B, F, -1)
    idx_c = idx_c.view(1, 1, -1).expand(B, F, -1)
    idx_d = idx_d.view(1, 1, -1).expand(B, F, -1)

    # Use gather
    a = torch.gather(flat_integral, 2, idx_a)
    b = torch.gather(flat_integral, 2, idx_b)
    c = torch.gather(flat_integral, 2, idx_c)
    d = torch.gather(flat_integral, 2, idx_d)

    output_b =  (a - b - c + d).view(B, F, idx_shape[0], idx_shape[1])
    output = output_b 
    # print("output shape", output.shape, "output_b shape", output_b.shape)
    # if torch.allclose(output, output_b):
    #     print("output and output_b are the same")

    return output



def retrieval_K_2D_integral_image(integral_image, x1, y1, x2, y2):
    # integral_image: B, F, D+1, H+1, W+1
    B, F, H_, W_ = integral_image.shape
    # subvolume = integral_image[:,:,:, x2+1, y2+1, z2+1] - integral_image[:,:,:, x1, y2+1, z2+1] - integral_image[:,:,:, x2+1, y1, z2+1] - integral_image[:,:,:, x2+1, y2+1, z1] + integral_image[:,:,:, x1, y1, z2+1] + integral_image[:,:,:, x1, y2+1, z1] + integral_image[:,:,:, x2+1, y1, z1] - integral_image[:,:,:, x1, y1, z1]
    subvolume = integral_image[:,:,x2+1,y2+1] - integral_image[:,:,x1,y2+1] - integral_image[:,:,x2+1,y1] + integral_image[:,:,x1,y1]
    return subvolume


def retrieval_KV_2D_integral_image(integral_image, x1, y1, x2, y2):
    # integral_image: B, F, F, D+1, H+1, W+1
    # retrieval_loc: Nx6, 6 for x1, y1, z1, x2, y2, z2
    # output: B x F x F x (x2 - x1 + 1) x (y2 - y1 + 1) x (z2 - z1 + 1)
    B, F, F, H_, W_ = integral_image.shape
    subvolume = integral_image[:,:,:,x2+1,y2+1] - integral_image[:,:,:,x1,y2+1] - integral_image[:,:,:,x2+1,y1] + integral_image[:,:,:,x1,y1]
    return subvolume


# @torch.compile
def build_2D_k_integral_image(k):
    """
    # k: B, F, H, W
    # output: B, F, H+1, W+1
    """
    B, F,  H, W = k.shape

    # k_pad = torch.nn.functional.pad(k, pad=(1,0,1,0))
    # print("k type", k.dtype)
    output = torch.cumsum(k, dim=2) # .to(k.dtype)
    # print("output cumsum 1type", output.dtype)
    output = torch.cumsum(output, dim=3) # .to(k.dtype)
    # print("output cumsum 2type", output.dtype)
    output = torch.nn.functional.pad(output, pad=(1,0,1,0))


    return output# .contiguous()


# @torch.compile
def build_2D_kv_integral_image(k, v):
    """
    # k: B, F, H, W
    # v: B, F, H, W
    # output: B, F, F, H+1, W+1
    """

    t0 = time()
    assert k.shape[-2:] == v.shape[-2:]
    B, F, H, W = k.shape
    _, F_, H, W = v.shape

    kv_product = torch.einsum('bfhw,bghw->bfghw', k, v) # .to(k.dtype) # pair wise
    # print("kv_product shape", kv_product.shape, "kv_product dtype", kv_product.dtype)

    kv_product = torch.cumsum(kv_product, dim=3) # .to(k.dtype)
    kv_product = torch.cumsum(kv_product, dim=4) # .to(k.dtype)
    # pad zero to make it B, F, F, H+1, W+1
    kv_product = torch.nn.functional.pad(kv_product, pad=(1,0,1,0))
    return kv_product
    # ([512, 768, 768, 14, 14])


def build_convolve_kernel_2D_from_window_size(window_size, device='cuda'):
    # window_size: dxhxw
    # output kernel: dxhxw
    window_size = torch.tensor(window_size)
    h, w = window_size
    kernel = torch.zeros(2,2,device=device)
    kernel[1, 1] = 1.
    kernel[1, 0] = -1.
    kernel[0, 1] = -1.
    kernel[0, 0] = 1.

    dialation_size = window_size 

    return kernel, dialation_size




def cumsum_tester():
    # Example input
    B = 128
    embed = 768
    heads_dim = 64
    heads = embed // heads_dim
    H = 14
    W = 14

    b = B 
    h = heads
    d = heads_dim // 4
    d_ = heads_dim 

    qk_embed = d * heads  # d * heads = embed // 3
    v_embed = d_ * heads  # d_ * heads = embed

    H_ = H 
    W_ = W

    steps = 100
    inception_attn = 'integral_map_conv2d'
    # amp_type = 'bfloat16'  # 'bfloat16' or 'float16' # float32

    window_size = (14, 14)
    linear_layer = torch.nn.Linear(embed, (2*qk_embed + v_embed), bias=False)  # B, embed -> B, embed*3
    linear_layer = linear_layer.to(device='cuda', dtype=torch.bfloat16)

    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=1e-3)
    Loss = torch.nn.MSELoss() 

    for step_i in range(steps):
        t0 = time()
        # ok, if we do not want to use whole window, can we get a smaller intergral map?
        
        # all datatype to bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.float16): # bfloat16 float32
            x = torch.randn(B, embed, H, W, device='cuda', dtype=torch.float16, requires_grad=False)  # B, embed, H, W
            target = torch.randn(B, embed, H, W, device='cuda', dtype=torch.float16, requires_grad=False)  # B, embed, H, W
            x_1d = x.permute(0, 2, 3, 1).reshape(B , H * W, embed)  # B, H, W, embed -> B*H*W, embed
            qkv_map = linear_layer(x_1d)  # B, embed, H, W
            qkv_map = qkv_map.reshape(B, H, W,(2*qk_embed + v_embed)).permute(0, 3, 1, 2)  # B, embed, H, W

            q = qkv_map[:, :qk_embed, :, :]  # B, qk_embed, H, W
            k = qkv_map[:, qk_embed:qk_embed + qk_embed, :, :]  # B, qk_embed, H, W
            v = qkv_map[:, qk_embed + qk_embed:, :, :]  # B, v_embed, H, W

            # q = torch.randn(B*heads, heads_dim, H, W, device='cuda', dtype=torch.bfloat16)
            # k = torch.randn(B*heads, heads_dim, H, W, device='cuda', dtype=torch.bfloat16)
            # v = torch.randn(B*heads, heads_dim, H, W, device='cuda', dtype=torch.bfloat16)

            q = torch.sigmoid(q)  # Simulate some processing
            k = torch.sigmoid(k)  # Simulate some processing
            v = torch.sigmoid(v)  # Simulate some processing

            # to head
            q = rearrange(q, 'b (n d) h w -> (b n) d h w', n=heads, d=d)  # B, heads, heads_dim, H, W
            k = rearrange(k, 'b (n d) h w -> (b n) d h w', n=heads, d=d)  # B, heads, heads_dim, H, 
            v = rearrange(v, 'b (n d) h w -> (b n) d h w', n=heads, d=d_)  # B, heads, heads_dim, H, W

            q_2d = q 
            k_2d = k
            v_2d = v

        if inception_attn == 'integral_map_conv2d':
            intergral_kv_map = build_2D_kv_integral_image(k, v)
            t1 = time()
            intergral_k_map = build_2D_k_integral_image(k)
            t2 = time()


            print("intergral_kv_map shape", intergral_kv_map.shape, "gpu memory take", intergral_kv_map.element_size() * intergral_kv_map.nelement() / 1024**2, "MB")
            print("intergral_k_map shape", intergral_k_map.shape, "gpu memory take", intergral_k_map.element_size() * intergral_k_map.nelement() / 1024**2, "MB")

            # intergral_kv_map = intergral_kv_map.reshape(b*h,d*d_,H_+1,W_+1) # 
            # intergral_k_map = intergral_k_map.reshape(b*h,d,H_+1,W_+1)
            intergral_kv_map = intergral_kv_map.reshape(b*h*d*d_, 1,H_+1,W_+1) # 
            intergral_k_map = intergral_k_map.reshape(b*h*d,1,H_+1,W_+1)


            kernel, dialation_size = build_convolve_kernel_2D_from_window_size(window_size, device=x.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.to(intergral_kv_map.device).to(intergral_kv_map.dtype)
            kernel.requires_grad = False
            dialation_size = dialation_size 

            # check if kernel is over 2^31 -1 

            kernel_kv = kernel
            kernel_k = kernel
            


            # kernel_kv = kernel.repeat(d*d_, 1, 1, 1)  # B, F, F, -1
            # kernel_k = kernel.repeat(d, 1, 1, 1)  # B, F, -1

            # window_kv = torch.nn.functional.conv2d(input=intergral_kv_map, weight=kernel_kv, dilation=tuple(dialation_size), padding=0, stride=1)
            # window_k = torch.nn.functional.conv2d(input=intergral_k_map, weight=kernel_k, dilation=tuple(dialation_size), padding=0, stride=1) 

            # print("investigate why conv2d slow", "output window_kv", window_kv.shape, "output window_k", window_k.shape,
            #     "intergral_kv_map shape", intergral_kv_map.shape, "intergral_kv_map dtype", intergral_kv_map.dtype,
            #     "intergral_k_map shape", intergral_k_map.shape, "intergral_k_map dtype", intergral_k_map.dtype,
            #     "kernel shape", kernel.shape, "kernel dtype", kernel.dtype,
            #     "dilation size", dialation_size,
            # )
            # raise Exception("investigate why conv2d slow")

        
            # t3 = time()

            # target_shape = torch.tensor([H_,W_])
            # window_kv_shape = torch.tensor(window_kv.shape[-2:])
            # pad_h = (target_shape[0] - window_kv_shape[0]) // 2
            # pad_h_end = target_shape[0] - window_kv_shape[0] - pad_h
            # pad_w = (target_shape[1] - window_kv_shape[1]) // 2
            # pad_w_end = target_shape[1] - window_kv_shape[1] - pad_w

            # pad_operation = nn.ReplicationPad2d((pad_w, pad_w_end, pad_h, pad_h_end))
            # t4 = time()
            # # print("pad details", pad_w, pad_w_end, pad_h, pad_h_end) # 3 4 3 4 
            # window_kv_pad = pad_operation(window_kv)
            # window_k_pad = pad_operation(window_k)       

            # window_kv_pad = window_kv_pad.view(b*h, d, d_, H_, W_) # B, F, F, -1
            # window_k_pad = window_k_pad.view(b*h, d, H_, W_) # B, F, -1     
            
            # q_2d B, d, H, W
            attn = torch.einsum('bdhw,bdfhw->bfhw',q_2d, window_kv_pad)
            norm = torch.einsum('bdhw,bdhw->bhw',q_2d, window_k_pad)
            # print("attn", attn.dtype, "norm", norm.dtype)
            attn = attn / (norm.unsqueeze(1) + 1e-6)
            # print("norm attn", attn.dtype)
            attn = attn.reshape(b,h,d_,H*W)
            # print("attn", attn.shape)
            attn_2d = attn.reshape(b, h* d_, H, W)  # B, heads, heads_dim, H, W 
            print(
                "attn_2d shape", attn_2d.shape, "attn_2d dtype", attn_2d.dtype,
            )
        elif inception_attn == 'conv2d':
            kv_product = torch.einsum('bfhw,bghw->bfghw', k_2d, v_2d).to(k.dtype)
            B_, F0, F1, _, _ = kv_product.shape
            kv_product = kv_product.reshape(B_,F0*F1, H_, W_)
            kernel = torch.ones(window_size).to(kv_product.device).to(kv_product.dtype) 
            kernel.requires_grad = False
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            window_kv_product_ = torch.nn.functional.avg_pool2d(input=kv_product, kernel_size=window_size, stride=1, padding=0)
            window_k_prodct_ = torch.nn.functional.avg_pool2d(input=k_2d, kernel_size=window_size, stride=1, padding=0)
            window_kv_product = window_kv_product_*(window_size[0]*window_size[1])
            window_k_prodct = window_k_prodct_*(window_size[0]*window_size[1])
            target_shape = torch.tensor([H_,W_])
            window_kv_shape = torch.tensor(window_kv_product.shape[-2:])
            pad_h = (target_shape[0] - window_kv_shape[0]) // 2
            pad_h_end = target_shape[0] - window_kv_shape[0] - pad_h
            pad_w = (target_shape[1] - window_kv_shape[1]) // 2
            pad_w_end = target_shape[1] - window_kv_shape[1] - pad_w
        
            pad_operation = nn.ReplicationPad2d((pad_w, pad_w_end, pad_h, pad_h_end))
            # print("pad details", pad_w, pad_w_end, pad_h, pad_h_end) # 3 4 3 4 
            window_kv_product = pad_operation(window_kv_product)
            window_k_prodct = pad_operation(window_k_prodct)     


        t5 = time()
        forward_time = t5 - t1
        loss = Loss(attn_2d, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t6 = time()
        backward_time = t6 - t5

        print(f"Step {step_i+1}/{steps}, Forward Time: {forward_time:.4f}s, Backward Time: {backward_time:.4f}s")
    
        print("Detail time", "kv integral map build time", t1-t0, "k integral map build time", t2-t1, "intergral map conv time", t3-t2, "pad operation time", t4-t3,)


if __name__ == "__main__":
    cumsum_tester()
    # print("cumsum_tester completed")



"""
float32
intergral_kv_map shape torch.Size([6144, 64, 64, 15, 15]) gpu memory take 21600.0 MB
intergral_k_map shape torch.Size([6144, 64, 15, 15]) gpu memory take 337.5 MB

bfloat16
intergral_kv_map shape torch.Size([6144, 64, 64, 15, 15]) gpu memory take 10800.0 MB
RuntimeError: output gradient tensor must fit into 32-bit index math

bfloat16 q // 4
intergral_kv_map shape torch.Size([6144, 16, 64, 15, 15]) gpu memory take 2700.0 MB
intergral_k_map shape torch.Size([6144, 16, 15, 15]) gpu memory take 42.1875 MB




Detail time kv integral map build time 0.0005996227264404297 k integral map build time 3.814697265625e-05 intergral map conv time 22.56050753593445 pad operation time 0.0001506805419921875





"""