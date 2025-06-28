"""
InceptionNeXt paper: https://arxiv.org/abs/2303.16900
Original implementation & weights from: https://github.com/sail-sg/inceptionnext
"""

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import trunc_normal_, DropPath, to_2tuple, get_padding, SelectAdaptivePool2d
from ._builder import build_model_with_cfg
from ._features import feature_take_indices
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

__all__ = ['HieraInceptFormer']


from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from time import time
# helpers
# from einops import einsum



class MlpClassifierHead(nn.Module):
    """ MLP classification head
    """

    def __init__(
            self,
            in_features,
            num_classes=1000,
            pool_type='avg',
            mlp_ratio=3,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop=0.,
            bias=True
    ):
        super().__init__()
        self.use_conv = False
        self.in_features = in_features
        self.num_features = hidden_features = int(mlp_ratio * in_features)

        assert pool_type, 'Cannot disable pooling'
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)

        self.fc1 = nn.Linear(in_features * self.global_pool.feat_mult(), hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def reset(self, num_classes: int, pool_type: Optional[str] = None):
        if pool_type is not None:
            assert pool_type, 'Cannot disable pooling'
            self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=True)

        self.fc2 = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        return x if pre_logits else self.fc2(x)



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
# @torch.compile
def build_2D_k_integral_image(k, pad=True):
    """
    # k: B, F, H, W
    # output: B, F, H+1, W+1
    """
    # k_pad = torch.nn.functional.pad(k, pad=(1,0,1,0))
    # print("k type", k.dtype)
    output = torch.cumsum(k, dim=2) # .to(k.dtype)
    # print("output cumsum 1type", output.dtype)
    output = torch.cumsum(output, dim=3) # .to(k.dtype)
    # print("output cumsum 2type", output.dtype)
    if pad:
        output = torch.nn.functional.pad(output, pad=(1,0,1,0))

    return output# .contiguous()



# @torch.compile
def build_2D_kv_integral_image(k, v, pad=True):
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
    if pad:
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


def build_convolve_kernel_2D_from_multi_window_size(window_size, device='cuda'):
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



class InceptionAttention(nn.Module):
    def __init__(self,  dim, 
                        heads = 8, 
                        qk_dim_compression = 1,
                        kv_token_num_compression=0, 
                        dropout = 0., 
                        windows=[0], 
                        qk_nonlin='sigmoid', 
                        inception_attn='integral_map_conv2d', 
                        inception_merge='cat', 
                        do_padding=False, 
                        ):
        super().__init__()
        inner_dim = dim 
        self.windows = windows
        self.heads = heads
        self.inception_attn = inception_attn
        self.kv_token_num_compression = kv_token_num_compression
        self.qk_nonlin = qk_nonlin

        if qk_dim_compression > 1:
            qk_inner_dim = dim // qk_dim_compression
        else:
            qk_inner_dim = dim

        assert inception_attn in ['integral_map_conv2d','conv2d'], f"Unknown inception type {inception_attn}"     
        if self.kv_token_num_compression > 1:
            self.maxpool2d = nn.MaxPool2d(kernel_size=kv_token_num_compression, stride=kv_token_num_compression, padding=0)
            self.upsample2d = nn.Upsample(scale_factor=kv_token_num_compression, mode='bilinear', align_corners=False)

        self.to_q = torch.nn.Conv2d(dim, qk_inner_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.to_k = torch.nn.Conv2d(dim, qk_inner_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.to_v = torch.nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0, bias=False)

        assert inception_merge in ['mean', 'cat','linear']
        self.inception_merge = inception_merge
        self.post_padding = do_padding
        self.out_in_dim = inner_dim * len(windows) 
        self.to_out = nn.Sequential(
            nn.Conv2d(self.out_in_dim, dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout)
        ) 

    def forward(self, x):
        "x: B, C, H, W"

        B, C, H, W = x.shape
        assert H == W, f"so far only support square input, got H={H}, W={W}"

        h = self.heads
        q = self.to_q(x) # B, C, H, W
        k = self.to_k(x) # B, C, H, W
        v = self.to_v(x) # B, C, H, W

        q = rearrange(q, 'b (h d) H W -> (b h) d H W', h = self.heads)
        k = rearrange(k, 'b (h d) H W -> (b h) d H W', h = self.heads)
        v = rearrange(v, 'b (h d) H W -> (b h) d H W', h = self.heads)

        if self.kv_token_num_compression > 1:
            k = self.maxpool2d(k) # B, H, W, d
            v = self.maxpool2d(v) # B, H, W, d

        if self.qk_nonlin == 'elu':
            q = torch.nn.functional.elu(q) + 1 + 1e-6
            k = torch.nn.functional.elu(k) + 1 + 1e-6


        elif self.qk_nonlin == 'sigmoid':
            q = torch.sigmoid(q) 
            k = torch.sigmoid(k) 
            # normllize the v
            # we need to normalize v if float16
            v = torch.nn.functional.normalize(v, dim=1, p=2) # B, H, W, d
 
        bh, d, _, _ = q.shape # q have H and W
        bh_, d_, H_, W_ = v.shape

        def divide_by_two_until_odd(x):
            if x == 0:
                return 0
            while x % 2 == 0:
                x = x // 2
            return x

        Inception_window_size = []
        for window in self.windows:
            if window > 0 and H_ >= window and W_ >= window:
                Inception_window_size.append([H_//window, W_//window])

        assert len(Inception_window_size) > 0 or self.windows == [0], f"At least one window size should be provided, got {self.windows}"
        # if len(Inception_window_size) == 0:
        #     # all proposed window not work, find the window which 
        #     min_H = divide_by_two_until_odd(H)
        #     min_W = divide_by_two_until_odd(W)
        #     Inception_window_size = [[min_H, min_W]]

        if self.windows == [0]:
            k = k.view(B,h,d,H_*W_).permute(0,1,3,2)
            v = v.view(B,h,d_,H_*W_).permute(0,1,3,2) # B, H, N, F
            q = q.view(B,h, d,H*W).permute(0,1,3,2) # B, H, N, F
            context = k.transpose(-2, -1) @ v # B, H, N, F
            k_sum = k.sum(dim=-2, keepdim=True) # B, H, N, 1
            attn = q @ context # B, H, N, F
            Z =  q @ k_sum.transpose(-2, -1) + 1e-6 # B, H, N, 1
            attn = attn / Z # B, H, N, F
            attn = rearrange(attn, 'b h n d -> b (h d) n')
            attn = attn.view(B, C, H, W) # B, C, H, W
            # norm inf 
            if torch.isnan(attn).any():
                # early NaN
                print("out has nan")
                print("norm", Z.min(),Z.max())
                print("attn", attn.min(), attn.max())
                print("q_2d", q.min(), q.max())
                print("k_sum", k_sum.min(), k_sum.max())
                print("context", context.min(), context.max())
                raise ValueError("Stop here due to NaN")
        


            return self.to_out(attn)

        if self.inception_attn == 'integral_map_conv2d':
            intergral_k_map = build_2D_k_integral_image(k)  # B, F, H+1, W+1
            intergral_kv_map = build_2D_kv_integral_image(k, v) # B, F, F, H+1, W+1
            intergral_k_map = rearrange(intergral_k_map, 'b f h w -> (b f) 1 h w') # B, F, H+1, W+1
            intergral_kv_map = rearrange(intergral_kv_map, 'b f e h w -> (b f e) 1 h w') # B, F, F, H+1, W+1


        if self.inception_merge == 'linear':
            pass 
        else:
            output_attn_list = []

            for window_size in Inception_window_size:
                kernel, dialation_size = build_convolve_kernel_2D_from_window_size(window_size, device=x.device)
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                kernel = kernel.to(intergral_kv_map.device).to(intergral_kv_map.dtype)
                kernel.requires_grad = False
                dialation_size = dialation_size 


                if not self.post_padding:
                    # calculate padding ourselves
                    actual_kernel_size = kernel.shape[-1] + dialation_size[0] -1 
                    target_pad = actual_kernel_size - 2
                    target_pad_right = target_pad // 2
                    target_pad_left = target_pad - target_pad_right
                    # print("before padding intergral_kv_map", intergral_kv_map.shape, "intergral_k_map", intergral_k_map.shape, "actual_kernel_size", actual_kernel_size)
                    intergral_kv_map_pad = torch.nn.functional.pad(intergral_kv_map, (target_pad_left, target_pad_right, target_pad_left, target_pad_right), mode='replicate')
                    intergral_k_map_pad = torch.nn.functional.pad(intergral_k_map, (target_pad_left, target_pad_right, target_pad_left, target_pad_right), mode='replicate')

                    window_kv = torch.nn.functional.conv2d(input=intergral_kv_map_pad, weight=kernel, dilation=tuple(dialation_size), stride=1)
                    window_k = torch.nn.functional.conv2d(input=intergral_k_map_pad, weight=kernel, dilation=tuple(dialation_size), stride=1) 

                    # print("intergral_kv_map", intergral_kv_map.shape, "window_kv", window_kv.shape)
                    # print("intergral_k_map", intergral_k_map.shape, "window_k", window_k.shape)

                    window_kv = window_kv.view(bh, d, d_, H_, W_) # B, F, F, -1
                    window_k = window_k.view(bh, d, H_, W_) # B, F, -1

                else:
                    window_kv = torch.nn.functional.conv2d(input=intergral_kv_map, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1) # B, 1, H+1-w, W+1-w
                    window_k = torch.nn.functional.conv2d(input=intergral_k_map, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1) 

                    target_shape = torch.tensor([H_,W_])
                    window_kv_shape = torch.tensor(window_kv.shape[-2:])
                    pad_h = (target_shape[0] - window_kv_shape[0]) // 2
                    pad_h_end = target_shape[0] - window_kv_shape[0] - pad_h
                    pad_w = (target_shape[1] - window_kv_shape[1]) // 2
                    pad_w_end = target_shape[1] - window_kv_shape[1] - pad_w
                    
                    pad_operation = nn.ReplicationPad2d((pad_w, pad_w_end, pad_h, pad_h_end)) # no replication need, how to do 
                    # print("pad details", pad_w, pad_w_end, pad_h, pad_h_end) # 3 4 3 4 
                    window_kv = pad_operation(window_kv)
                    window_k = pad_operation(window_k)       

                    window_kv = window_kv.view(bh, d, d_, H_, W_) # B, F, F, -1
                    window_k = window_k.view(bh, d, H_, W_) # B, F, -1

                attn = torch.einsum('bdhw,bdfhw->bfhw',q, window_kv)
                norm = torch.einsum('bdhw,bdhw->bhw',q, window_k)
                attn = attn / (norm.unsqueeze(1) + 1e-6)
                
                attn = attn.reshape(B, C, H, W) # B, C, H, W
                if torch.isnan(attn).any():
                    # early NaN
                    print("out has nan")
                    print("norm", norm.min(),norm.max())
                    print("attn", attn.min(), attn.max())
                    print("q_2d", q.min(), q.max())
                    print("window_kv", window_kv.min(), window_kv.max())
                    print("window_k", window_k.min(), window_k.max())
                    raise ValueError("Stop here due to NaN")
            


                # (b,h*d_,H,W)
                output_attn_list.append(attn)
                        
            if self.inception_merge == 'mean':
                output_attn = torch.stack(output_attn_list, dim=0).mean(dim=0)
            else:
                output_attn = torch.cat(output_attn_list, dim=1) 
        t_iter_end = time()
        # out = rearrange(output_attn, 'b h n d -> b n (h d)')
        if output_attn.shape[1] != self.out_in_dim:
            # print("Check inception windows", Inception_window_size, "self.windows", self.windows, "X", x.shape)
            # pad to same channel
            output_attn = torch.nn.functional.pad(output_attn, (0, 0, 0, 0, 0, self.out_in_dim - output_attn.shape[1]))

        return self.to_out(output_attn)

class HieraInceptFormerStage(nn.Module):
    def __init__(self, 
                dim, 
                mlp_dim,
                heads, 
                qk_dim_compression,
                kv_token_num_compression,
                dropout, 
                windows,  
                qk_nonlin,
                inception_attn, 
                inception_merge, 
                do_padding,
                ):
        super().__init__()
        self.norm_layer_0 = torch.nn.BatchNorm2d(dim)
        norm_layer_1 = torch.nn.BatchNorm2d(dim)
        self.attn = InceptionAttention(dim, 
                                        heads=heads, 
                                        qk_dim_compression=qk_dim_compression,
                                        kv_token_num_compression=kv_token_num_compression,
                                        dropout=dropout, 
                                        windows=windows,  
                                        qk_nonlin=qk_nonlin,
                                        inception_attn=inception_attn, 
                                        inception_merge=inception_merge, 
                                        do_padding=do_padding,
                                        )
       
        self.FeedForward = torch.nn.Sequential(
            norm_layer_1,
            torch.nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            torch.nn.Conv2d(mlp_dim, dim, kernel_size=1,),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x = self.attn(self.norm_layer_0(x)) + x
        x = self.FeedForward(x) + x
        return x




class HieraInceptFormer(nn.Module):
    r""" HieraInceptFormer
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt` - https://arxiv.org/abs/2303.16900

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
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            drop_rate=0.,
            drop_path_rate=0.,
            windows=[(0,), (0,), (0,), (0,)],
            inception_attn='conv2d', 
            inception_merge='mean', 
            do_padding=False, 
            qk_dim_compression=(0, 0, 0, 0), 
            kv_token_num_compression=(0, 0, 0, 0),
            qk_nonlin='sigmoid',

    ):
        super().__init__()

        num_stage = len(depths)

        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.drop_rate = drop_rate
        self.feature_info = []

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.Sequential()

        for i in range(num_stage):
            out_chs = dims[i]
            for block_depth in range(depths[i]):
                self.stages.append(HieraInceptFormerStage(
                    dim=out_chs,
                    mlp_dim=out_chs * mlp_ratios[i],
                    heads= out_chs // 96,  # heads = dim // 64
                    qk_dim_compression= qk_dim_compression[i],
                    kv_token_num_compression= kv_token_num_compression[i],
                    dropout= dp_rates[i][block_depth] if isinstance(dp_rates[i], list) else dp_rates[i],
                    windows= windows[i],
                    qk_nonlin = qk_nonlin,
                    inception_attn = inception_attn,
                    inception_merge = inception_merge,
                    do_padding = do_padding,
                ))
            if i < num_stage - 1:  # no downsample on last stage
                self.stages.append(nn.Conv2d(out_chs, dims[i+1], kernel_size=3, stride=2, padding=1))
            prev_chs = out_chs
            self.feature_info += [dict(num_chs=prev_chs, module=f'stages.{i}')]

        self.num_features = prev_chs
        self.head_drop = nn.Dropout(drop_rate)
        # self.head = nn.Linear(
        #     prev_chs, num_classes) if num_classes > 0 else nn.Identity()
        self.head = MlpClassifierHead(self.num_features, num_classes, pool_type=self.global_pool, drop=drop_rate)
        self.head_hidden_size = self.head.num_features

        # self.head_hidden_size = self.num_features
    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
            ]
        )

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc2

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages), indices)

        # forward pass
        x = self.stem(x)
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index + 1]

        for feat_idx, stage in enumerate(stages):
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.stages), indices)
        self.stages = self.stages[:max_index + 1]  # truncate blocks w/ stem as idx 0
        if prune_head:
            self.reset_classifier(0, 'avg')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        # x = self.stages(x)
        for layer_num, layer in enumerate(self.stages):
            x = layer(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)


    def forward(self, x):
        x = self.forward_features(x)
        # if x.shape[2] == self.num_features and len(x.shape) == 4:
        #     x = rearrange(x, 'b c h w -> b (h w) c')  # flatten for head
        #     x = x.mean(dim=1)  # global pool

        x = self.forward_head(x)
        # if NaN detected, stop
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc2',
        **kwargs
    }


# default_cfgs = generate_default_cfgs({
#     'inception_next_atto.sail_in1k': _cfg(
#         hf_hub_id='timm/',
#         # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_atto.pth',
#     ),
#     'inception_next_tiny.sail_in1k': _cfg(
#         hf_hub_id='timm/',
#         # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth',
#     ),
#     'inception_next_small.sail_in1k': _cfg(
#         hf_hub_id='timm/',
#         # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth',
#     ),
#     'inception_next_base.sail_in1k': _cfg(
#         hf_hub_id='timm/',
#         # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth',
#         crop_pct=0.95,
#     ),
#     'inception_next_base.sail_in1k_384': _cfg(
#         hf_hub_id='timm/',
#         # url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base_384.pth',
#         input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
#     ),
# })


def _create_hiera_inception_former(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        HieraInceptFormer, variant, pretrained,
        **kwargs,
    )
    return model




@register_model
def hiera_inception_former_tiny_w0(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
        windows=[[2,4,8],[2,4,8],[1,2,4],[0]]
    )
    return _create_hiera_inception_former('hiera_inception_former_tiny', pretrained=pretrained, **dict(model_args, **kwargs))


# @register_model
# def inception_next_small(pretrained=False, **kwargs):
#     model_args = dict(
#         depths=(3, 3, 27, 3), dims=(96, 192, 384, 768),
#         token_mixers=InceptionDWConv2d,
#     )
#     return _create_inception_next('inception_next_small', pretrained=pretrained, **dict(model_args, **kwargs))


# @register_model
# def inception_next_base(pretrained=False, **kwargs):
#     model_args = dict(
#         depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
#         token_mixers=InceptionDWConv2d,
#     )
#     return _create_inception_next('inception_next_base', pretrained=pretrained, **dict(model_args, **kwargs))
