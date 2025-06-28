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
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs
from timm.layers import (
    Attention,
    AttentionPoolLatent,
    PatchEmbed,
    Mlp,
    SwiGLUPacked,
    SwiGLU,
    LayerNorm,
    RmsNorm,
    DropPath,
    PatchDropout,
    trunc_normal_,
    lecun_normal_,
    resample_patch_embed,
    resample_abs_pos_embed,
    use_fused_attn,
    get_act_layer,
    get_norm_layer,
    maybe_add_mask,
    LayerType,
)

from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import named_apply, checkpoint, checkpoint_seq, adapt_input_conv
from ._registry import generate_default_cfgs, register_model, register_model_deprecations

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


@torch.compile
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


@torch.compile
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

    output = torch.cumsum(kv_product, dim=3) # .to(k.dtype)
    output = torch.cumsum(output, dim=4) # .to(k.dtype)
    # pad zero to make it B, F, F, H+1, W+1
    output = torch.nn.functional.pad(output, pad=(1,0,1,0))
    return output


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



class LinearInceptionAttention(nn.Module):
    def __init__(self, dim, heads = 8, qk_dim_compression=False, kv_token_num_compression=False, dropout = 0., windows=[0], qk_nonlin='elu', inception_attn='integral_map_indices', inception_merge='cat', do_padding=False, debug=False):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        if qk_dim_compression:
            qk_dim_head = dim_head // 2
        else:
            qk_dim_head = dim_head
        qk_inner_dim = qk_dim_head * heads

        project_out = not (heads == 1 and dim_head == dim)
        assert inception_attn in ['integral_map_indices','integral_map_conv2d', 'integral_map_sum', 'conv2d'], f"Unknown inception type {inception_attn}"
        self.inception_attn = inception_attn
        self.kv_token_num_compression = kv_token_num_compression
        if self.kv_token_num_compression:
            self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.upsample2d = nn.Upsample(scale_factor=2, mode='nearest')
        self.debug = debug

        self.windows = windows
        self.heads = heads


        self.to_q = nn.Linear(dim, qk_inner_dim, bias = False)
        self.to_k = nn.Linear(dim, qk_inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)



        assert inception_merge in ['mean', 'cat']
        self.inception_merge = inception_merge
        self.do_padding = do_padding

        if self.inception_merge == 'cat':
            inception_dim = inner_dim * len(windows)
        else:
            inception_dim = inner_dim
  
        self.to_out = nn.Sequential(
            nn.Linear(inception_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        


        self.qk_nonlin = qk_nonlin
        # Inception_window_size = [[8,8]]#[[3,3],[5,5],[7,7]]


    def forward(self, x, debug=False, attn_mask=None,):

        t0 = time()
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)

        if self.kv_token_num_compression:
            b, h, n, d = k.shape
            k_2d = rearrange(k, 'b h (H W) d -> (b h) d H W', H=int(np.sqrt(k.shape[2])), W=int(np.sqrt(k.shape[2])))
            k_2d_comp = self.maxpool2d(k_2d)
            v_2d = rearrange(v, 'b h (H W) d -> (b h) d H W', H=int(np.sqrt(v.shape[2])), W=int(np.sqrt(v.shape[2])))
            v_2d_comp = self.maxpool2d(v_2d)

            k = rearrange(k_2d_comp, '(b h) d H W -> b h (H W) d', b=b, h=h, H=k_2d_comp.shape[2], W=k_2d_comp.shape[2])
            v = rearrange(v_2d_comp, '(b h) d H W -> b h (H W) d', b=b, h=h, H=v_2d_comp.shape[2], W=v_2d_comp.shape[2])

        if self.qk_nonlin == 'elu':
            q = torch.nn.functional.elu(q) + 1 + 1e-6
            k = torch.nn.functional.elu(k) + 1 + 1e-6

        elif self.qk_nonlin == 'elu_clip':
            q = torch.nn.functional.elu(q) + 1 + 1e-6
            k = torch.nn.functional.elu(k) + 1 + 1e-6
            max_type_val = torch.finfo(q.dtype).max
            cumsum_nums = H*W 
            clip_value = (max_type_val / cumsum_nums)
            q = torch.clip(q, min=-clip_value, max=clip_value)    
            k = torch.clip(k, min=-clip_value, max=clip_value)
        
        elif self.qk_nonlin == 'elu_log':
            # elu log will autocast to float32
            # using another torch.log on the > 0 value
            q = torch.nn.functional.elu(q) 
            k = torch.nn.functional.elu(k)
            q_positive_mask = q > 0
            k_positive_mask = k > 0
            q = torch.where(q_positive_mask, torch.log(q + 1), q)
            k = torch.where(k_positive_mask, torch.log(k + 1), k)
            q = q + 1
            k = k + 1

        elif self.qk_nonlin == 'softmax':                    
            q = q.softmax(dim=-1) # softmax make to float32
            k = k.softmax(dim=-2) # on the number of features

        b,h,n,d = q.shape
        H,W = int(np.sqrt(n)), int(np.sqrt(n))
        _,_,n_,d_ = v.shape
        H_,W_ = int(np.sqrt(n_)), int(np.sqrt(n_))
        assert H*W == n
        # # 0 global size, 1 small size 2x2, 2 medium size 4x4 3: large size 6x6 4: mixed size 2x2, 4x4, 6x6

        def divide_by_two_until_odd(x):
            if x == 0:
                return 0
            while x % 2 == 0:
                x = x // 2
            return x


        Inception_window_size = []
        for window in self.windows:
            if window > 0 and H % window == 0 and W % window == 0: 
                Inception_window_size.append([H_//window, W_//window])
        if len(Inception_window_size) == 0:
            # all proposed window not work, find the window which 
            min_H = divide_by_two_until_odd(H)
            min_W = divide_by_two_until_odd(W)
            Inception_window_size = [[min_H, min_W]]


        q_2d = rearrange(q, 'b h (H W) d -> (b h) d H W', H=H, W=W)
        k_2d = rearrange(k, 'b h (H W) d -> (b h) d H W', H=H_, W=W_)
        v_2d = rearrange(v, 'b h (H W) d -> (b h) d H W', H=H_, W=W_)

        if self.debug or self.windows == [0]:
            # k: B, H, N, F 
            # k_2d: B*H, F, H, W 
            k_1d = k_2d.view(b,h,d,H_*W_).permute(0,1,3,2) # B, H, N, F
            v_1d = v_2d.view(b,h,d_,H_*W_).permute(0,1,3,2) # B, H, N, F
            context_debug = k_1d.transpose(-2, -1) @ v_1d
            k_sum_debug = k_1d.sum(dim=-2, keepdim=True) 
            q_1d = q_2d.view(b,h,d,H*W).permute(0,1,3,2) # B, H, N, F
            # if self.kv_token_num_compression:
            #     context_debug_2d = context_debug.permute(0, 1, 3, 2).view(b*h, d_, H_//2, W_//2) # B*H, F, H, W
            #     context_debug_2d_upsample = self.upsample2d(context_debug_2d) # B*H, F, H*2, W*2
            #     context_debug = rearrange(context_debug_2d_upsample, '(b h) d H W -> b h (H W) d', b=b, h=h, H=H_, W=W_)
            #     k_sum_debug_2d = k_sum_debug.permute(0, 1, 3, 2).view(b*h, 1, H_//2, W_//2) # B*H, 1, H, W
            #     k_sum_debug_2d_upsample = self.upsample2d(k_sum_debug_2d) # B*H, 1, H*2, W*2
            #     k_sum_debug = rearrange(k_sum_debug_2d_upsample, '(b h) d H W -> b h (H W) d', b=b, h=h, H=H_, W=W_)

            attn_debug = q_1d @ context_debug # B, H, N, F
            Z = q_1d @ k_sum_debug.transpose(-2, -1) + 1e-6 # B, H, N, 1
            out_debug = attn_debug / Z # B, H, N, F
            out_debug = rearrange(out_debug, 'b h n d -> b n (h d)')
            if self.windows == [0]:
                return self.to_out(out_debug)
            

            

        if self.inception_attn == 'integral_map_indices' or self.inception_attn == 'integral_map_conv2d' or self.inception_attn == 'integral_map_sum':
            # achieve sum of the contect and k locally by intergral map, slow in backward, recommand for large window size
            k_to_intergral_k = k_2d.clone()
            k_to_intergral_kv = k_2d.clone()
            v_to_intergral_kv = v_2d.clone()

            intergral_k_map = build_2D_k_integral_image(k_to_intergral_k)  # B, F, H+1, W+1 
            intergral_kv_map = build_2D_kv_integral_image(k_to_intergral_kv, v_to_intergral_kv) # B, F, F, H+1, W+1

            # if intergral_k_map.min() < 0:
            #     print("intergral_k_map has negative value")
            #     print("intergral_k_map", intergral_k_map.min(), intergral_k_map.max(), intergral_k_map.dtype())

            if self.debug or self.inception_attn == 'integral_map_sum':
                retrived_kv_2d = retrieval_KV_2D_integral_image(intergral_kv_map, 0, 0, H_-1, W_-1)
                retrived_k_2d = retrieval_K_2D_integral_image(intergral_k_map, 0, 0, H_-1, W_-1)
                retrived_kv_1d = retrived_kv_2d.view(b,h,d,d_)
                retrived_k_1d = retrived_k_2d.view(b,h,1,d)

            if self.debug :
                print("context_debug dtype", context_debug.dtype)
                print("retrived_kv_1d dtype", retrived_kv_1d.dtype)
                context_debug = context_debug.to(retrived_kv_1d.dtype)
                if not torch.allclose(context_debug, retrived_kv_1d):
                    minus_kv = torch.subtract(context_debug, retrived_kv_1d).abs()
                    minus_kv = minus_kv.float()
                    print("context != window_kv", minus_kv.shape) # torch.Size([256, 8, 64, 64]) torch.Size([256, 8, 64, 64])
                    print("diff min", minus_kv.min(),
                            'diff mean', minus_kv.mean(), 
                            'diff max', minus_kv.max())
                    min_idx_multi = torch.unravel_index(minus_kv.argmin(), minus_kv.shape)
                    print("min index", min_idx_multi, 'value', minus_kv[min_idx_multi])
                    max_idx_multi = torch.unravel_index(minus_kv.argmax(), minus_kv.shape)
                    print("max index", max_idx_multi, 'value', minus_kv[max_idx_multi])
                    print("in context_debug", context_debug[min_idx_multi])
                    print("in retrived_kv_1d", retrived_kv_1d[min_idx_multi])
                    print("minus again", (context_debug[min_idx_multi] - retrived_kv_1d[min_idx_multi]).abs())
                else:
                    print("context == window_kv")

                if not torch.allclose(k_sum_debug, retrived_k_1d):
                    print("k_sum_debug dtype", k_sum_debug.dtype)
                    print("retrived_k_1d dtype", retrived_k_1d.dtype)
                    k_sum_debug = k_sum_debug.to(retrived_k_1d.dtype)
                    minus_k = torch.subtract(k_sum_debug , retrived_k_1d).abs()
                    minus_k = minus_k.float()
                    print("k_sum != window_k", minus_k.shape)
                    print("diff min", minus_k.min(),
                            'min index', minus_k.argmin(),
                            'diff mean', minus_k.mean(), 
                            'diff max',  minus_k.max(),
                            'max index', minus_k.argmax())
                    min_idx_multi = torch.unravel_index(minus_k.argmin(), minus_k.shape)
                    print("min index", min_idx_multi, 'value', minus_k[min_idx_multi])
                    max_idx_multi = torch.unravel_index(minus_k.argmax(), minus_k.shape)
                    print("max index", max_idx_multi, 'value', minus_k[max_idx_multi])
                    print("in k_sum_debug", k_sum_debug[min_idx_multi])
                    print("in retrived_k_1d", retrived_k_1d[min_idx_multi])
                    print("minus again", (k_sum_debug[min_idx_multi] - retrived_k_1d[min_idx_multi]).abs())
                else:
                    print("k_sum == window_k")
            
                intergral_k_map = intergral_k_map.reshape(b*h*d,1,H_+1,W_+1)
                for window_size in Inception_window_size:
                    window_k = multi_retrieval_2D_integral_image(intergral_k_map, window_size)
                    # check if negative value exist in the window_k
                    if window_k.min() < 0:
                        print("window_k has negative value")
                        print("window_k", window_k.min(), window_k.max())

                raise ValueError("Stop here")
            
            if self.inception_attn == 'integral_map_sum':
                """
                mimic the linear attention, get global kv
                """
                q_1d = q_2d.view(b,h,d,H*W).permute(0,1,3,2)
                attn = q_1d @ retrived_kv_1d # (batch_size, heads, token_length, dim_head)
                clip_value = torch.finfo(x.dtype).max - 1
                attn = torch.clip(attn, min=-clip_value, max=clip_value)

                Z = q_1d @ retrived_k_1d.transpose(-2, -1) + 1e-6 # (batch_size, heads, token_length, 1)
                out = attn / Z # (batch_size, heads, token_length, dim_head)
        
                out = rearrange(out, 'b h n d -> b n (h d)')
                # print("out", out.shape, out.dtype)

                return self.to_out(out)                
            intergral_kv_map = intergral_kv_map.reshape(b*h*d*d_,1,H_+1,W_+1) # 
            intergral_k_map = intergral_k_map.reshape(b*h*d,1,H_+1,W_+1)

        output_attn_list = []

        for window_size in Inception_window_size:
            if self.inception_attn == 'integral_map_indices':
   
                intergral_kv_map = intergral_kv_map.reshape(b*h,d*d_,H_+1,W_+1)   # 
                intergral_k_map = intergral_k_map.reshape(b*h,d,H_+1,W_+1)

                window_kv = multi_retrieval_2D_integral_image(intergral_kv_map, window_size)
                window_k = multi_retrieval_2D_integral_image(intergral_k_map, window_size)

                window_kv = window_kv.view(b*h, d, d_, H, W) # B, F, F, -1
                window_k = window_k.view(b*h, d, H, W) # B, F, -1        
                # q_2d B, d, H, W
                attn = torch.einsum('bdhw,bdfhw->bfhw',q_2d, window_kv)
                norm = torch.einsum('bdhw,bdhw->bhw',q_2d, window_k)

                max_type_val = torch.finfo(attn.dtype).max - 1
                attn = torch.clip(attn, min=-max_type_val, max=max_type_val)
                norm = torch.clip(norm, min=1e-6, max=max_type_val)
                attn_norm = attn / (norm.unsqueeze(1) + 1e-6)
                attn_norm = attn_norm.clip(min=-max_type_val, max=max_type_val)
                attn_norm = attn_norm.reshape(b,h,d_,H*W)
                output_attn_list.append(attn_norm.permute(0,1,3,2))

                if torch.isnan(attn).any():
                    print("out has nan")
                    print("norm", norm.min(),norm.max())
                    print("attn_norm", attn_norm.min(), attn_norm.max())
                    print("q_2d", q_2d.min(), q_2d.max())
                    print("window_kv_product", window_kv.min(), window_kv.max())
                    print("window_k_prodct", window_k.min(), window_k.max())
                    raise ValueError("Stop here due to NaN")
            
                # print("window_kv", window_kv.shape, window_kv.dtype)
                # print("window_k", window_k.shape, window_k.dtype)
                # raise ValueError("Stop here")
                
            elif self.inception_attn == 'integral_map_conv2d':
                kernel, dialation_size = build_convolve_kernel_2D_from_window_size(window_size, device=x.device)
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                kernel = kernel.to(intergral_kv_map.device).to(intergral_kv_map.dtype)
                kernel.requires_grad = False
                dialation_size = dialation_size 

                if self.do_padding:
                    window_kv = torch.nn.functional.conv2d(input=intergral_kv_map, weight=kernel, dilation=tuple(dialation_size), padding='same', stride=1)
                    window_k = torch.nn.functional.conv2d(input=intergral_k_map, weight=kernel, dilation=tuple(dialation_size), padding='same', stride=1) 
                    window_kv_pad = window_kv[:,:,:H_,:W_] # for even size kernel, no need to pad
                    window_k_pad = window_k[:,:,:H_,:W_]
                    window_kv_pad = window_kv_pad.view(b*h, d, d_, H_, W_) # B, F, F, -1
                    window_k_pad = window_k_pad.view(b*h, d, H_, W_) # B, F, -1

                else:
                    window_kv = torch.nn.functional.conv2d(input=intergral_kv_map, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)
                    window_k = torch.nn.functional.conv2d(input=intergral_k_map, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1) 

                    # print("window_kv", window_kv.shape, window_kv.dtype) # conv2d give float16
                    # print("window_k", window_k.shape, window_k.dtype) # conv2d give float16
                    target_shape = torch.tensor([H_,W_])
                    window_kv_shape = torch.tensor(window_kv.shape[-2:])
                    pad_h = (target_shape[0] - window_kv_shape[0]) // 2
                    pad_h_end = target_shape[0] - window_kv_shape[0] - pad_h
                    pad_w = (target_shape[1] - window_kv_shape[1]) // 2
                    pad_w_end = target_shape[1] - window_kv_shape[1] - pad_w
                
                    pad_operation = nn.ReplicationPad2d((pad_w, pad_w_end, pad_h, pad_h_end))
                    # print("pad details", pad_w, pad_w_end, pad_h, pad_h_end) # 3 4 3 4 
                    window_kv_pad = pad_operation(window_kv)
                    window_k_pad = pad_operation(window_k)       

                    window_kv_pad = window_kv_pad.view(b*h, d, d_, H_, W_) # B, F, F, -1
                    window_k_pad = window_k_pad.view(b*h, d, H_, W_) # B, F, -1        
                # q_2d B, d, H, W
                attn = torch.einsum('bdhw,bdfhw->bfhw',q_2d, window_kv_pad)
                norm = torch.einsum('bdhw,bdhw->bhw',q_2d, window_k_pad)
                # print("attn", attn.dtype, "norm", norm.dtype)
                attn = attn / (norm.unsqueeze(1) + 1e-6)
                # print("norm attn", attn.dtype)
                attn = attn.reshape(b,h,d_,H*W)
                # output_attn = output_attn + attn.permute(0,1,3,2)
                output_attn_list.append(attn.permute(0,1,3,2))


            elif self.inception_attn == 'conv2d':
                # for window size we want odd number,find the odd number smaller than the window size
                # Inception_window_size = ensure_odd_window_sizes(Inception_window_size)
                # no integral map, work for small patches operation, very efficient
                kv_product = torch.einsum('bfhw,bghw->bfghw', k_2d, v_2d).to(k.dtype) # pair wise, B, F, F, H, W

                B_, F0, F1, _, _ = kv_product.shape
                kv_product = kv_product.reshape(B_,F0*F1, H_, W_)

                kernel = torch.ones(window_size).to(kv_product.device).to(kv_product.dtype) 
                kernel.requires_grad = False
                kernel = kernel.unsqueeze(0).unsqueeze(0)
        
                #  choice 0 let's try padding version first
                if self.do_padding:
                    window_kv_product = torch.nn.functional.avg_pool2d(input=kv_product, kernel_size=window_size, stride=1, padding=[window_size[0]//2, window_size[1]//2]) *(window_size[0]*window_size[1])
                    window_k_prodct = torch.nn.functional.avg_pool2d(input=k_2d, kernel_size=window_size, stride=1, padding=[window_size[0]//2, window_size[1]//2]) *(window_size[0]*window_size[1])

                    window_kv_product = window_kv_product[:,:,:H_,:W_] # for even size kernel, no need to pad
                    window_k_prodct = window_k_prodct[:,:,:H_,:W_]
                    
                    window_kv_product = window_kv_product.view(B_, F0, F1, H_, W_)

                else:    
                    window_kv_product_ = torch.nn.functional.avg_pool2d(input=kv_product, kernel_size=window_size, stride=1, padding=0)
                    window_k_prodct_ = torch.nn.functional.avg_pool2d(input=k_2d, kernel_size=window_size, stride=1, padding=0)
                    window_kv_product = window_kv_product_*(window_size[0]*window_size[1])
                    window_k_prodct = window_k_prodct_*(window_size[0]*window_size[1])

                    # print("before padding window_kv_product", window_kv_product.shape, "window_k_prodct", window_k_prodct.shape)

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
                    # print("padded window_kv_product", window_kv_product.shape, "window_k_prodct", window_k_prodct.shape)

                    window_kv_product = window_kv_product.view(B_, F0, F1, H_, W_)

                if self.kv_token_num_compression:
                    # upsample
                    window_kv_product = window_kv_product.view(B_,F0*F1, H_, W_)
                    window_kv_product_upsample = self.upsample2d(window_kv_product) # B, F0*F1, H*2, W*2
                    window_k_prodct_upsample = self.upsample2d(window_k_prodct)

                    window_kv_product = window_kv_product_upsample.view(B_,F0,F1, H_*2, W_*2) # B, F0, F1, H*2, W*2
                    window_k_prodct = window_k_prodct_upsample.view(B_,F0, H_*2, W_*2) # B, F0, H*2, W*2


                attn = torch.einsum('bdhw,bdfhw->bfhw',q_2d, window_kv_product)
                # attn clip value
                # max_type_val = torch.finfo(attn.dtype).max - 1
                # attn = torch.clip(attn, min=-max_type_val, max=max_type_val)
                norm = torch.einsum('bdhw,bdhw->bhw',q_2d, window_k_prodct)
                # norm = torch.clip(norm, min=1e-6, max=max_type_val)
                attn_norm = attn / (norm.unsqueeze(1) + 1e-6)
                # attn_norm = attn_norm.clip(min=-max_type_val, max=max_type_val)

                attn_norm = attn_norm.reshape(b,h,d_,H*W)
                output_attn_list.append(attn_norm.permute(0,1,3,2))

                if torch.isnan(attn).any():
                    print("out has nan")
                    print("norm", norm.min(),norm.max())
                    print("attn_norm", attn_norm.min(), attn_norm.max())
                    print("q_2d", q_2d.min(), q_2d.max())
                    print("window_kv_product", window_kv_product.min(), window_kv_product.max())
                    print("window_k_prodct", window_k_prodct.min(), window_k_prodct.max())
                    raise ValueError("Stop here due to NaN")
            


        if self.inception_merge == 'mean':
            output_attn = torch.stack(output_attn_list, dim=0).mean(dim=0)
        else:
            output_attn = torch.cat(output_attn_list, dim=-1) # (b, h, n, d_*N)
        t_iter_end = time()
        out = rearrange(output_attn, 'b h n d -> b n (h d)')


        return self.to_out(out)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)




class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()

        # torch.autograd.set_detect_anomaly(True)
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        inv_freq = inv_freq.to(dtype=torch.float64)
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """

        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)

        # print("emb_x", emb_x.shape, emb_x.min(), emb_x.max())
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc




class InceptionFormerBlock(nn.Module):
    """Transformer block with pre-normalization."""
    """Include one feed forward, attn blocks."""
    

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            windows: Tuple[int, int] = (1,),
            qk_dim_compression: bool = False,
            kv_token_num_compression: bool = False,
            qk_nonlin: str = 'sigmoid',
            inception_attn: str = 'integral_map_sum',
            inception_merge: str = 'cat',
            do_padding: bool = False,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LinearInceptionAttention(
            dim,
            heads=num_heads,
            qk_dim_compression=qk_dim_compression,
            kv_token_num_compression=kv_token_num_compression,
            dropout=attn_drop,
            windows=windows,
            qk_nonlin=qk_nonlin,
            inception_attn=inception_attn,
            inception_merge=inception_merge,
            do_padding=do_padding,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x


class InceptFormer(nn.Module):
    r""" InceptFormer

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalization layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            scale_attn_norm: bool = False,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = False,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            pool_include_prefix: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = InceptionFormerBlock,
            mlp_layer: Type[nn.Module] = Mlp,
            # inception params
            qk_nonlin='sigmoid',
            qk_dim_compression=False,
            kv_token_num_compression=False,
            do_padding=False,
            inception_attn='integral_map_sum', 
            inception_merge='cat',
            windows: Tuple[int, int] = (1,),
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or LayerNorm
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            InceptionFormerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_attn_norm=scale_attn_norm,
                scale_mlp_norm=scale_mlp_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                windows = windows,
                qk_dim_compression=qk_dim_compression,
                kv_token_num_compression=kv_token_num_compression,
                qk_nonlin=qk_nonlin,
                inception_attn=inception_attn,
                inception_merge=inception_merge,
                do_padding=do_padding,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()


    def fix_init_weight(self) -> None:
        """Apply weight initialization fix (scaling w/ layer index)."""
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        """Initialize model weights.

        Args:
            mode: Weight initialization mode ('jax', 'jax_nlhb', 'moco', or '').
        """
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        # named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for a single module (compatibility method)."""
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        """Load pretrained weights.

        Args:
            checkpoint_path: Path to checkpoint.
            prefix: Prefix for state dict keys.
        """
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Set of parameters that should not use weight decay."""
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict[str, Union[str, List]]:
        """Create regex patterns for parameter grouping.

        Args:
            coarse: Use coarse grouping.

        Returns:
            Dictionary mapping group names to regex patterns.
        """
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
        """
        self.grad_checkpointing = enable
        if hasattr(self.patch_embed, 'set_grad_checkpointing'):
            self.patch_embed.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """Get the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Reset the classifier head.

        Args:
            num_classes: Number of classes for new classifier.
            global_pool: Global pooling type.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_input_size(
            self,
            img_size: Optional[Tuple[int, int]] = None,
            patch_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Update the input image resolution and patch size.

        Args:
            img_size: New input resolution, if None current resolution is used.
            patch_size: New patch size, if None existing patch size is used.
        """
        prev_grid_size = self.patch_embed.grid_size
        self.patch_embed.set_input_size(img_size=img_size, patch_size=patch_size)
        if self.pos_embed is not None:
            num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
            num_new_tokens = self.patch_embed.num_patches + num_prefix_tokens
            if num_new_tokens != self.pos_embed.shape[1]:
                self.pos_embed = nn.Parameter(resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=self.patch_embed.grid_size,
                    old_size=prev_grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    verbose=True,
                ))

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            output_dict: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            output_dict: Return outputs as a dictionary with 'image_features' and 'image_intermediates' keys
            attn_mask: Optional attention mask for masked attention (e.g., for NaFlex)
        Returns:
            A tuple with (final_features, intermediates), a list of intermediate features, or a dictionary containing
            'image_features' and 'image_intermediates' (and optionally 'image_intermediates_prefix')
        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # forward pass
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x, attn_mask=attn_mask)
            if i in take_indices:
                # normalize intermediates with final norm layer if enabled
                intermediates.append(self.norm(x) if norm else x)

        # process intermediates
        if self.num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:self.num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, self.num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None

        if reshape:
            # reshape to BCHW output format
            H, W = self.patch_embed.dynamic_feat_size((height, width))
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        # For dictionary output, handle prefix tokens separately
        if output_dict:
            result_dict = {}
            # Intermediates are always included
            result_dict['image_intermediates'] = intermediates
            if prefix_tokens is not None and return_prefix_tokens:
                result_dict['image_intermediates_prefix'] = prefix_tokens

            # Only include features if not intermediates_only
            if not intermediates_only:
                x_final = self.norm(x)
                result_dict['image_features'] = x_final

            return result_dict

        # For non-dictionary output, maintain the original behavior
        if not torch.jit.is_scripting() and return_prefix_tokens and prefix_tokens is not None:
            # return_prefix not support in torchscript due to poor type handling
            intermediates = list(zip(intermediates, prefix_tokens))

        if intermediates_only:
            return intermediates

        x = self.norm(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ) -> List[int]:
        """Prune layers not required for specified intermediates.

        Args:
            indices: Indices of intermediate layers to keep.
            prune_norm: Whether to prune normalization layer.
            prune_head: Whether to prune the classifier head.

        Returns:
            List of indices that were kept.
        """
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)
        self.blocks = self.blocks[:max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, List[int], Tuple[int]] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Get intermediate layer outputs (DINO interface compatibility).

        NOTE: This API is for backwards compat, favour using forward_intermediates() directly.

        Args:
            x: Input tensor.
            n: Number or indices of layers.
            reshape: Reshape to NCHW format.
            return_prefix_tokens: Return prefix tokens.
            norm: Apply normalization.

        Returns:
            List of intermediate features.
        """
        return self.forward_intermediates(
            x, n,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            output_fmt='NCHW' if reshape else 'NLC',
            intermediates_only=True,
            attn_mask=attn_mask,
        )

    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        """Apply pooling to feature tokens.

        Args:
            x: Feature tensor.
            pool_type: Pooling type override.

        Returns:
            Pooled features.
        """
        if self.attn_pool is not None:
            if not self.pool_include_prefix:
                x = x[:, self.num_prefix_tokens:]
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass through classifier head.

        Args:
            x: Feature tensor.
            pre_logits: Return features before final classifier.

        Returns:
            Output tensor.
        """
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x


def _create_inception_former(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        InceptFormer,
        variant,
        pretrained,
        **kwargs
    )
    return model


@register_model
def inceptionformer_base_patch16_224(pretrained: bool = False, **kwargs) -> InceptFormer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_inception_former('inceptionformer_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model



@register_model
def inceptionformer_tiny_patch16_224(pretrained: bool = False, **kwargs) -> InceptFormer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=512, depth=6, num_heads=8)
    model = _create_inception_former('inceptionformer_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

