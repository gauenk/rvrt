# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
import warnings
import math
import torch
import torch as th
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange
from ..op.deform_attn import deform_attn, DeformAttnPack


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5]):
        super(SpyNet, self).__init__()
        self.return_levels = return_levels
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if not os.path.exists(load_path):
                import requests
                url = 'https://github.com/JingyunLiang/RVRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading SpyNet pretrained model from {url}')
                os.makedirs(os.path.dirname(load_path), exist_ok=True)
                open(load_path, 'wb').write(r.content)

            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        # print("[a] ref[0].shape: ",ref[0].shape)
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        # print("ref[0].shape: ",ref[0].shape)

        for level in range(5):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        # print("ref[0].shape: ",ref[0].shape)
        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])
        # print(flow.shape)

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(input=flow, scale_factor=2,
                                           mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

            if level in self.return_levels:
                scale = 2 ** (5 - level)  # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                flow_out = F.interpolate(input=flow, size=(h // scale, w // scale), mode='bilinear',
                                         align_corners=False)
                flow_out[:, 0, :, :] *= float(w // scale) / float(w_floor // scale)
                flow_out[:, 1, :, :] *= float(h // scale) / float(h_floor // scale)
                flow_list.insert(0, flow_out)

        return flow_list

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(input=ref, size=(h_floor, w_floor), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_floor, w_floor), mode='bilinear', align_corners=False)

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


def get_fixed_offsets(ws2,ngroups,fixed_offset_max,shape,device):
    ws = int(math.sqrt(ws2))
    fixed = th.arange(ws) - ws//2
    fixed = fixed * fixed_offset_max
    mesh = th.stack(th.meshgrid(fixed,fixed),0).flatten(1,2)
    mesh = mesh.T.flatten()
    mesh = mesh.repeat(ngroups).flatten()
    nlocs = len(mesh)
    b,n,_,nH,nW = shape
    mesh = mesh.reshape((1,1,nlocs,1,1)).float().to(device)
    mesh = mesh.repeat((b,n,1,nH,nW))
    return mesh,mesh

def remove_time(dists,inds,t):

    # print(dists.shape)
    dshape = dists.shape
    ishape = inds.shape

    # -- filter --
    dists = dists.view(-1,1)
    inds = inds.view(-1,3)
    args = th.where(inds[...,0] != t)

    # -- filter dists --
    dists = th.gather(dists,1,args)

    # -- filter inds --
    inds_r = []
    for i in range(3):
        # print(i,inds[...,i].shape)
        inds_r.append(th.gather(inds[...,i],1,args))
    inds_r = th.stack(inds_r)

    # -- shape --

    return dists,inds

def get_search_offests(qvid,kvid,flows,k,ps,ws,stride1,dist_type,nheads):

    # -- info --
    # print("qvid.shape: ",qvid.shape)
    # print("kvid.shape: ",kvid.shape)
    b,clipsize,nftrs,H,W = qvid.shape
    # fflow = th.stack(fflow)
    # print("fflow.shape: ",fflow.shape)
    # bflow = th.zeros_like(fflow)
    # print("bflow.shape: ",bflow.shape)

    # -- create offset for optical flow --
    # ivid = th.inf * th.ones_like(qvid[:,:1])
    # qvid = th.stack([ivid,qvid],1)
    # kvid = th.stack([kvid,ivid],1)
    # zflow = th.zeros_like(fflow[0])

    # -- search order to match rvrt --
    qorder = [0,1,1,0] # frames [0,1,1,0]
    korder = [0,1,0,1] # frames [2,3,2,3]
    forder = [[0,0],[0,1],[1,0],[1,1]]
    # from einops import rearrange

    # -- grid for normalize --
    # from einops import rearrange
    # device = qvid.device
    # dtype = qvid.dtype
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, H, dtype=dtype, device=device),
    #                                 torch.arange(0, W, dtype=dtype, device=device))
    # grid = torch.stack((grid_y, grid_x), 2).float()  # W(x), H(y), 2
    # grid = rearrange(grid,'H W two -> two H W').requires_grad_(False)

    # -- searching --
    import stnls
    # dist_type = "l2"
    search = stnls.search.init({"search_name":"paired",
                                "k":k,"ps":ps,"ws":ws,
                                "stride0":1,"stride1":stride1,
                                "self_action":"anchor",
                                "nheads":nheads,"dist_type":dist_type,
                                "itype":"float","full_ws":False})
    B = qvid.shape[0]
    qvid_n = th.cat([qvid[:,qi] for qi in qorder])
    kvid_n = th.cat([kvid[:,ki] for ki in korder])
    fflow_n = th.cat([flows[fi[0]][:,fi[1]] for fi in forder])[:,None]
    dists,inds = search(qvid_n,kvid_n,fflow_n)

    # -- prepare inds --
    inds = rearrange(inds,'b HD H W k two -> b (HD k) two H W')
    inds = inds - fflow_n.flip(-3).detach()
    inds = rearrange(inds,'(b ngroups) ... -> ngroups b ...',b=B)

    # -- extract --
    # print(inds.shape)
    shape_str = "(clipinfo) b HDk two H W -> "
    shape_str += "b clipinfo (HDk two) H W"
    inds = rearrange(inds,shape_str,H=H,W=W)

    # -- unpack for readability --
    # print(inds[0,0,:18,32,32].reshape(9,2))
    # ra = th.rand(1).item()
    # inds[0,0,:18,32,32] = ra
    # print(ra)
    offset1 = th.stack([inds[:,0],inds[:,1]],1)
    offset2 = th.stack([inds[:,2],inds[:,3]],1)

    # offset1 = th.stack([inds[:,0],inds[:,3]],1)
    # offset2 = th.stack([inds[:,1],inds[:,2]],1)

    # offset1 = th.stack([inds[:,0],inds[:,1]],1)
    # offset2 = th.stack([inds[:,3],inds[:,2]],1)

    return offset1,offset2


class GuidedDeformAttnPack(DeformAttnPack):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.offset_type = kwargs.pop('offset_type', "default")
        self.fixed_offset_max = kwargs.pop('fixed_offset_max', 2.5)
        self.offset_ps = kwargs.pop('offset_ps', 1)
        self.offset_ws = kwargs.pop('offset_ws', 7)
        self.offset_stride1 = kwargs.pop('offset_stride1', 0.5)
        self.offset_dtype = kwargs.pop('offset_dtype', "l2")
        # print(self.offset_type,self.offset_ws,self.offset_stride1,self.offset_dtype)
        # print(self.offset_type,self.offset_ws,self.offset_stride1)
        # self.offset_ws = kwargs.pop('offset_ws', 21)
        # self.offset_stride1 = kwargs.pop('offset_stride1', 0.05)

        super(GuidedDeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.in_channels * (1 + self.clip_size) + self.clip_size * 2, 64, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(64, self.clip_size * self.deformable_groups * self.attn_size * 2, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
        )
        self.conv_offset = None if self.offset_type != "default" else self.conv_offset
        self.init_offset()

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_k = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_v = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                  nn.Linear(self.proj_channels, self.in_channels),
                                  Rearrange('n d h w c -> n d c h w'))
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                                 Rearrange('n d h w c -> n d c h w'))

        # -- create shell for hooks --
        self.flow_shell = nn.Identity()
        self.q_shell = nn.Identity()
        self.k_shell = nn.Identity()
        self.o1_offset_shell = nn.Identity()
        self.o2_offset_shell = nn.Identity()

    def init_offset(self):
        if hasattr(self, 'conv_offset') and not(self.conv_offset is None):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, q, k, v, v_prop_warped, flows, return_updateflow):

        # -- projection --
        b, t, c, h, w = q.shape
        proj_q = self.proj_q(q)
        proj_k = self.proj_k(k)
        proj_v = self.proj_v(v)

        # print("proj_q.shape: ",proj_q.shape,"proj_k.shape: ",proj_k.shape)
        kv = torch.cat([proj_k, proj_v], 2)

        # -- offsets --
        if self.offset_type == "default":
            offset1, offset2 = torch.chunk(self.max_residue_magnitude * torch.tanh(
                self.conv_offset(torch.cat([q] + v_prop_warped + flows, 2)\
                                 .transpose(1, 2)).transpose(1, 2)), 2, dim=2)
            # offset1 = offset1 + flows[0].flip(2).repeat(1, 1, offset1.size(2) // 2, 1, 1)
            # offset2 = offset2 + flows[1].flip(2).repeat(1, 1, offset2.size(2) // 2, 1, 1)
        elif self.offset_type == "fixed":
            offset1,offset2 = get_fixed_offsets(self.attn_size,
                                                self.deformable_groups,
                                                self.fixed_offset_max,
                                                q.shape,q.device)
        elif self.offset_type == "search":
            K = self.attn_size
            nheads = self.deformable_groups
            # strangely the "value" is the "key" here.
            offset1,offset2 = get_search_offests(proj_q,proj_k,flows,K,
                                                 self.offset_ps,self.offset_ws,
                                                 self.offset_stride1,
                                                 self.offset_dtype,nheads)
        else:
            offset1 = self.max_residue_magnitude * torch.randn_like(offset1).clamp(-1,1)
            offset2 = self.max_residue_magnitude * torch.randn_like(offset1).clamp(-1,1)
            # offset1 = flows[0].flip(2).repeat(1, 1, offset1.size(2) // 2, 1, 1)
            # offset2 = flows[1].flip(2).repeat(1, 1, offset2.size(2) // 2, 1, 1)
        # print(self.max_residue_magnitude)

        # -- added for hooks --
        # q = self.q_shell(proj_q)
        # k = self.k_shell(proj_k)
        # flows = self.flow_shell(flows)
        # offset1 = self.o1_offset_shell(offset1)
        # offset2 = self.o2_offset_shell(offset2)
        # print(offset1.shape,q.shape,flows[0].shape)

        # -- add optical flow --
        if self.offset_type in ["default","fixed","search"]:
            offset1 = offset1 + flows[0].flip(2).repeat(1, 1, offset1.size(2) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(2).repeat(1, 1, offset2.size(2) // 2, 1, 1)

        # -- cat --
        offset = torch.cat([offset1, offset2], dim=2).flatten(0, 1)
        # print(offset.shape,self.clip_size,self.deformable_groups,self.attn_size,
        #       flows[0].shape)
        # offset1 = offset1*0.
        # offset2 = offset2*0.
        # offset = offset * 0. # TODO: DELETE ME; testing only.

        # -- deform --
        b, t, c, h, w = offset1.shape
        # q = self.proj_q(q).view(b * t, 1, self.proj_channels, h, w)
        proj_q = proj_q.view(b * t, 1, self.proj_channels, h, w)
        # kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(proj_q, kv, offset, self.kernel_h, self.kernel_w, self.stride,
                        self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups,
                        self.clip_size).view(b, t, self.proj_channels, h, w)
        v = self.proj(v)
        v = v + self.mlp(v)

        if return_updateflow:
            return v, offset1.view(b, t, c // 2, 2, h, w).mean(2).flip(2), offset2.view(b, t, c // 2, 2, h, w).mean(
                2).flip(2)
        else:
            return v


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """ Window based multi-head self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, -1, dtype=q.dtype)  # Don't use attn.dtype after addition!
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index


class STL(nn.Module):
    """ Swin Transformer Layer (STL).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class STG(nn.Module):
    """ Swin Transformer Group (STG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[2, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            STL(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB).

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, **kwargs):
        super(RSTB, self).__init__()
        self.input_resolution = kwargs['input_resolution']

        self.residual_group = STG(**kwargs)
        self.linear = nn.Linear(kwargs['dim'], kwargs['dim'])

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4)


class RSTBWithInputConv(nn.Module):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1,
                 groups=1, num_blocks=2, **kwargs):
        super().__init__()

        main = []
        main += [Rearrange('n d c h w -> n c d h w'),
                 nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups),
                 Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n c d h w')]

        # RSTB blocks
        kwargs['use_checkpoint_attn'] = kwargs.pop('use_checkpoint_attn')[0]
        kwargs['use_checkpoint_ffn'] = kwargs.pop('use_checkpoint_ffn')[0]
        main.append(make_layer(RSTB, num_blocks, **kwargs))

        main += [Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n d c h w')]

        self.main = nn.Sequential(*main)

    def forward(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """
        return self.main(x)


class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.PixelShuffle(2))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.PixelShuffle(3))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class RVRT(nn.Module):
    """ Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT).
            A PyTorch impl of : `Recurrent Video Restoration Transformer with Guided Deformable Attention`  -
              https://arxiv.org/pdf/2205.00000

        Args:
            upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
            clip_size (int): Size of clip in recurrent restoration transformer.
            img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
            window_size (int | tuple(int)): Window size. Default: (2,8,8).
            num_blocks (list[int]): Number of RSTB blocks in each stage.
            depths (list[int]): Depths of each RSTB.
            embed_dims (list[int]): Number of linear projection output channels.
            num_heads (list[int]): Number of attention head of each stage.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
            qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
            inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
            spynet_path (str): Pretrained SpyNet model path.
            deformable_groups (int): Number of deformable groups in deformable attention. Default: 12.
            attention_heads (int): Number of attention heads in deformable attention. Default: 12.
            attention_window (list[int]): Attention window size in aeformable attention. Default: [3, 3].
            nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
            use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
            use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
            no_checkpoint_attn_blocks (list[int]): Layers without torch.checkpoint for attention modules.
            no_checkpoint_ffn_blocks (list[int]): Layers without torch.checkpoint for feed-forward modules.
            cpu_cache_length: (int): Maximum video length without cpu caching. Default: 100.
        """

    def __init__(self,
                 upscale=4,
                 clip_size=2,
                 img_size=[2, 64, 64],
                 window_size=[2, 8, 8],
                 num_blocks=[1, 2, 1],
                 depths=[2, 2, 2],
                 embed_dims=[144, 144, 144],
                 num_heads=[6, 6, 6],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 inputconv_groups=[1, 1, 1, 1, 1, 1],
                 spynet_path=None,
                 max_residue_magnitude=10,
                 deformable_groups=12,
                 attention_heads=12,
                 attention_window=[3, 3],
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 cpu_cache_length=100,
                 offset_type="default",
                 fixed_offset_max=2.5,
                 offset_ws=3,
                 offset_ps=1,offset_stride1=.5,offset_dtype="l2",
                 ):

        super().__init__()
        self.upscale = upscale
        self.clip_size = clip_size
        self.nonblind_denoising = nonblind_denoising
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in range(100)]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in range(100)]
        self.cpu_cache_length = cpu_cache_length
        self.times = {}

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # shallow feature extraction
        if self.upscale == 4:
            # video sr
            self.feat_extract = RSTBWithInputConv(in_channels=3,
                                                  kernel_size=(1, 3, 3),
                                                  groups=inputconv_groups[0],
                                                  num_blocks=num_blocks[0],
                                                  dim=embed_dims[0],
                                                  input_resolution=[1, img_size[1], img_size[2]],
                                                  depth=depths[0],
                                                  num_heads=num_heads[0],
                                                  window_size=[1, window_size[1], window_size[2]],
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  norm_layer=norm_layer,
                                                  use_checkpoint_attn=[False],
                                                  use_checkpoint_ffn=[False]
                                                  )
        else:
            # video deblurring/denoising
            self.feat_extract = nn.Sequential(Rearrange('n d c h w -> n c d h w'),
                                              nn.Conv3d(4 if self.nonblind_denoising else 3, embed_dims[0], (1, 3, 3),
                                                        (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              nn.Conv3d(embed_dims[0], embed_dims[0], (1, 3, 3), (1, 2, 2), (0, 1, 1)),
                                              nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                              Rearrange('n c d h w -> n d c h w'),
                                              RSTBWithInputConv(
                                                                in_channels=embed_dims[0],
                                                                kernel_size=(1, 3, 3),
                                                                groups=inputconv_groups[0],
                                                                num_blocks=num_blocks[0],
                                                                dim=embed_dims[0],
                                                                input_resolution=[1, img_size[1], img_size[2]],
                                                                depth=depths[0],
                                                                num_heads=num_heads[0],
                                                                window_size=[1, window_size[1], window_size[2]],
                                                                mlp_ratio=mlp_ratio,
                                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                norm_layer=norm_layer,
                                                                use_checkpoint_attn=[False],
                                                                use_checkpoint_ffn=[False]
                                                               )
                                              )

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        # recurrent feature refinement
        self.backbone = nn.ModuleDict()
        self.deform_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            # deformable attention
            self.deform_align[module] = GuidedDeformAttnPack(embed_dims[1],
                                                             embed_dims[1],
                                                             attention_window=attention_window,
                                                             attention_heads=attention_heads,
                                                             deformable_groups=deformable_groups,
                                                             clip_size=clip_size,
                                                             max_residue_magnitude=max_residue_magnitude,
                                                             offset_type=offset_type,
                                                             fixed_offset_max=fixed_offset_max,
                                                             offset_ws=offset_ws,
                                                             offset_ps=offset_ps,
                                                             offset_stride1=offset_stride1,
                                                             offset_dtype=offset_dtype)

            # feature propagation
            self.backbone[module] = RSTBWithInputConv(
                                                     in_channels=(2 + i) * embed_dims[0],
                                                     kernel_size=(1, 3, 3),
                                                     groups=inputconv_groups[i + 1],
                                                     num_blocks=num_blocks[1],
                                                     dim=embed_dims[1],
                                                     input_resolution=img_size,
                                                     depth=depths[1],
                                                     num_heads=num_heads[1],
                                                     window_size=window_size,
                                                     mlp_ratio=mlp_ratio,
                                                     qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                                                     norm_layer=norm_layer,
                                                     use_checkpoint_attn=[use_checkpoint_attns[i]],
                                                     use_checkpoint_ffn=[use_checkpoint_ffns[i]]
                                                     )

        # reconstruction
        self.reconstruction = RSTBWithInputConv(
                                               in_channels=5 * embed_dims[0],
                                               kernel_size=(1, 3, 3),
                                               groups=inputconv_groups[5],
                                               num_blocks=num_blocks[2],
                                               dim=embed_dims[2],
                                               input_resolution=[1, img_size[1], img_size[2]],
                                               depth=depths[2],
                                               num_heads=num_heads[2],
                                               window_size=[1, window_size[1], window_size[2]],
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               norm_layer=norm_layer,
                                               use_checkpoint_attn=[False],
                                               use_checkpoint_ffn=[False]
                                               )
        self.conv_before_upsampler = nn.Sequential(
                                                  nn.Conv3d(embed_dims[-1], 64, kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                                  )
        self.upsampler = Upsample(4, 64)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def propagate(self, feats, flows, module_name, updated_flows=None):
        """Propagate the latent clip features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, clip_size, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            updated_flows dict(list[tensor]): Each component is a list of updated
                optical flows with shape (n, clip_size, 2, h, w).

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()
        if 'backward' in module_name:
            flow_idx = range(0, t + 1)[::-1]
            clip_idx = range(0, (t + 1) // self.clip_size)[::-1]
        else:
            flow_idx = range(-1, t)
            clip_idx = range(0, (t + 1) // self.clip_size)

        if '_1' in module_name:
            updated_flows[f'{module_name}_n1'] = []
            updated_flows[f'{module_name}_n2'] = []

        # print("feats['shallow'][0].shape: ",feats['shallow'][0].shape)
        feat_prop = torch.zeros_like(feats['shallow'][0])
        if self.cpu_cache:
            feat_prop = feat_prop.cuda()

        # print(list(clip_idx))
        # print(feat_prop.shape)
        last_key = list(feats)[-2]
        fkey = list(feats.keys())
        # print(feats[fkey[0]].shape,flows[0].shape)
        # print("prop; ",len(clip_idx))
        for i in range(0, len(clip_idx)):
            idx_c = clip_idx[i]
            # print(i,clip_idx[i],clip_idx)
            if i > 0:
                if '_1' in module_name:
                    flow_n01 = flows[:, flow_idx[self.clip_size * i - 1], :, :, :]
                    flow_n12 = flows[:, flow_idx[self.clip_size * i], :, :, :]
                    flow_n23 = flows[:, flow_idx[self.clip_size * i + 1], :, :, :]
                    flow_n02 = flow_n12 + flow_warp(flow_n01, flow_n12.permute(0, 2, 3, 1))
                    flow_n13 = flow_n23 + flow_warp(flow_n12, flow_n23.permute(0, 2, 3, 1))
                    flow_n03 = flow_n23 + flow_warp(flow_n02, flow_n23.permute(0, 2, 3, 1))
                    flow_n1 = torch.stack([flow_n02, flow_n13], 1)
                    flow_n2 = torch.stack([flow_n12, flow_n03], 1)
                    if self.cpu_cache:
                        flow_n1 = flow_n1.cuda()
                        flow_n2 = flow_n2.cuda()
                else:
                    module_name_old = module_name.replace('_2', '_1')
                    flow_n1 = updated_flows[f'{module_name_old}_n1'][i - 1]
                    flow_n2 = updated_flows[f'{module_name_old}_n2'][i - 1]

                if self.cpu_cache:
                    if 'backward' in module_name:
                        feat_q = feats[last_key][idx_c].flip(1).cuda()
                        feat_k = feats[last_key][clip_idx[i - 1]].flip(1).cuda()
                    else:
                        feat_q = feats[last_key][idx_c].cuda()
                        feat_k = feats[last_key][clip_idx[i - 1]].cuda()
                else:
                    if 'backward' in module_name:
                        feat_q = feats[last_key][idx_c].flip(1)
                        feat_k = feats[last_key][clip_idx[i - 1]].flip(1)
                    else:
                        feat_q = feats[last_key][idx_c]
                        feat_k = feats[last_key][clip_idx[i - 1]]

                feat_prop_warped1 = flow_warp(feat_prop.flatten(0, 1),
                                           flow_n1.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
                feat_prop_warped2 = flow_warp(feat_prop.flip(1).flatten(0, 1),
                                           flow_n2.permute(0, 1, 3, 4, 2).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

                if '_1' in module_name:
                    feat_prop, flow_n1, flow_n2 = self.deform_align[module_name](
                        feat_q, feat_k, feat_prop,
                        [feat_prop_warped1, feat_prop_warped2],
                        [flow_n1, flow_n2], True)
                    updated_flows[f'{module_name}_n1'].append(flow_n1)
                    updated_flows[f'{module_name}_n2'].append(flow_n2)
                else:
                    feat_prop = self.deform_align[module_name](
                        feat_q, feat_k, feat_prop,
                        [feat_prop_warped1, feat_prop_warped2],
                        [flow_n1, flow_n2], False)
            # print(feat_prop.shape)

            if 'backward' in module_name:
                feat = [feats[k][idx_c].flip(1) for k in feats if k not in [module_name]] + [feat_prop]
            else:
                feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]


            if self.cpu_cache:
                feat = [f.cuda() for f in feat]
            # print([f.shape for f in feat])
            # print(module_name)
            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=2))
            # print("feat_prop.shape: ",feat_prop.shape)
            feats[module_name].append(feat_prop)
            # print(len(feats))

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            feats[module_name] = [f.flip(1) for f in feats[module_name]]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        feats['shallow'] = torch.cat(feats['shallow'], 1)
        feats['backward_1'] = torch.cat(feats['backward_1'], 1)
        feats['forward_1'] = torch.cat(feats['forward_1'], 1)
        feats['backward_2'] = torch.cat(feats['backward_2'], 1)
        feats['forward_2'] = torch.cat(feats['forward_2'], 1)
        # print(feats['backward_1'].shape)
        # print([(k,feats[k].shape) for k in feats])

        if self.cpu_cache:
            outputs = []
            for i in range(0, feats['shallow'].shape[1]):
                hr = torch.cat([feats[k][:, i:i + 1, :, :, :] for k in feats], dim=2)
                hr = self.reconstruction(hr.cuda())
                hr = self.conv_last(self.upsampler(
                    self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
                hr += torch.nn.functional.interpolate(lqs[:, i:i + 1, :, :, :].cuda(),
                                                      size=hr.shape[-3:],
                                                      mode='trilinear',
                                                      align_corners=False)
                hr = hr.cpu()
                outputs.append(hr)
                torch.cuda.empty_cache()

            return torch.cat(outputs, dim=1)

        else:
            hr = torch.cat([feats[k] for k in feats], dim=2)
            # print("hr.shape: ",hr.shape)
            hr = self.reconstruction(hr)
            # print("hr.shape: ",hr.shape)
            hr = self.conv_last(self.upsampler(
                self.conv_before_upsampler(hr.transpose(1, 2)))).transpose(1, 2)
            hr += torch.nn.functional.interpolate(lqs, size=hr.shape[-3:],
                                                  mode='trilinear', align_corners=False)

            return hr

    def forward(self, lqs, flows=None):
        """Forward function for RVRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        # -- optionally pad if testing --
        d_old = lqs.size(1)
        self.use_input_pad = not(self.training)
        if self.use_input_pad:
            d_pad = d_old % 2
            lqs = torch.cat([lqs, torch.flip(lqs[:, -d_pad:, ...], [1])], 1) \
                  if d_pad else lqs
        # print("lqs.shape:", lqs.shape,d_old,self.clip_size)

        # -- unpack --
        n, t, _, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.upscale == 4:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(lqs[:, :, :3, :, :].view(-1, 3, h, w),
                                           scale_factor=0.25, mode='bicubic')\
                              .view(n, t, 3, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        # print("clip_size: ",self.clip_size)
        # shallow feature extractions
        feats = {}
        if self.cpu_cache:
            feats['shallow'] = []
            # print("lqs.shape: ",lqs.shape)
            for i in range(0, t // self.clip_size):
                feat = self.feat_extract(lqs[:, i * self.clip_size:(i + 1) * self.clip_size, :, :, :]).cpu()
                # print("i, feat.shape: ",i, feat.shape)
                feats['shallow'].append(feat)
            flows_forward, flows_backward = self.compute_flow(lqs_downsample)

            lqs = lqs.cpu()
            lqs_downsample = lqs_downsample.cpu()
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
            torch.cuda.empty_cache()
        else:
            feats['shallow'] = list(torch.chunk(self.feat_extract(lqs), t // self.clip_size, dim=1))
            flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        # print(len(feats['shallow']),feats['shallow'][0].shape)

        # recurrent feature refinement
        updated_flows = {}
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(feats, flows, module_name, updated_flows)

        # reconstruction
        rec = self.upsample(lqs[:, :, :3, :, :], feats)

        # -- optionally slice --
        if self.use_input_pad:
            rec = rec[:,:d_old]

        return rec
