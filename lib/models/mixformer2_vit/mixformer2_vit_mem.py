import math
from functools import partial

import numpy as np
import torch
# torch.set_printoptions(profile="full")

import torch.nn as nn
import torch.nn.functional as F
import time

import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp, trunc_normal_
from torch.nn.init import xavier_uniform_
from lib.utils.misc import is_main_process
from lib.models.mixformer2_vit.head import build_box_head, build_score_decoder, build_score_decoder2, \
    build_mid_score_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.mixformer_vit.pos_util import get_2d_sincos_pos_embed

from einops import rearrange
from itertools import repeat
import collections.abc
import matplotlib.pyplot as plt


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.update = "delta"
        self.eps = 1e-8

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.betas = nn.Parameter(torch.randn(1, num_heads, 1, head_dim))
        self.proj_mem_k = nn.Linear(dim, dim,bias=qkv_bias)
        self.proj_mem_v = nn.Linear(dim, dim,bias=qkv_bias)

        self.qkv_mem = None
        self.k_mt = None
        self.v_mt = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #
        q, k, v = qkv.unbind(0)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w + 4], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w * 2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale  # 10,12,326,454
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w + 4, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_mem(self, x, t_h, t_w, s_h, s_w, mem, z):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w + 4], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w * 2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale  # 10,12,326,454
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v)

        sigma_q_s = nn.functional.elu(q_s) + 1.0
        att_mem = (sigma_q_s @ mem) / (sigma_q_s @ z)
        sigma_k_s = nn.functional.elu(k_s) + 1.0
        zz = sigma_k_s.sum(dim=-2, keepdim=True).transpose(-2, -1)

        if self.update == "linear":
            mem = mem + (sigma_k_s.transpose(-2, -1) @ v_s)

        #            mem = mem + sigma_k_s.transpose(-2, -1) @ v_s / zz
        elif self.update == "delta":
            mem = mem + \
                  sigma_k_s.transpose(-2, -1) @ (v_s - (sigma_k_s @ mem) / (sigma_k_s @ z))

        z = z + zz
        #        z = z + sigma_k_s.sum(dim=-2, keepdim=True).transpose(-2, -1)
        x_s = (torch.sigmoid(self.betas) * att_mem + (1 - torch.sigmoid(self.betas)) * x_s).transpose(1, 2).reshape(B,
                                                                                                                    s_h * s_w + 4,
                                                                                                                    C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, mem, z

    def forward_test(self, x, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        #        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 3, 1)
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, _, _ = qkv_s.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        #        q_s, k_s, v_s = qkv.unbind(0)
        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        _, k, v = qkv.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_online(self, x, t_h, t_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        #        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 3, 1)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dim = dim

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def forward_mem(self, x, t_h, t_w, s_h, s_w, mem, z):
        x, mem, z = self.attn.forward_mem(self.norm1(x), t_h, t_w, s_h, s_w, mem, z)
        x = x + self.drop_path1(x)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, mem, z

    def forward_test(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        x = x + self.drop_path1(self.attn.set_online(self.norm1(x), t_h, t_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, img_size_s=256, img_size_t=128, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None, reg_tokens_cfg=None):
        super(timm.models.vision_transformer.VisionTransformer, self).__init__()

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])
        self.feat_sz_s = img_size_s // patch_size
        self.feat_sz_t = img_size_t // patch_size
        self.num_patches_s = self.feat_sz_s ** 2
        self.num_patches_t = self.feat_sz_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)
        #        self.conv_ = nn.Conv2d(embed_dim,embed_dim,kernel_size=3,stride=2,padding=1,groups=dim,bias=False)

        self.reg_tokens = nn.Parameter(torch.randn(1, 4, embed_dim))
        self.pos_embed_reg = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # self.reg_tokens_cfg = reg_tokens_cfg
        self.init_pos_embed()

        # trunc_normal_(self.reg_tokens, std=.02)

    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                              cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))
        if is_main_process():
            print("Initialize pos embed with fixed sincos embedding.")

    def forward(self, x_t, x_ot, x_s, mem=None, z=None, addmem=True):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.feat_sz_s
        H_t = W_t = self.feat_sz_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t

        reg_tokens = self.reg_tokens.expand(B, -1, -1)  # (b, 4, embed_dim)
        reg_tokens = reg_tokens + self.pos_embed_reg

        x = torch.cat([x_t, x_ot, x_s, reg_tokens], dim=1)  # (b, hw+hw+HW+4, embed_dim)
        x = self.pos_drop(x)

        distill_feat_list = []
        early_search_tokens = []
        early_reg_tokens = []
        for i, blk in enumerate(self.blocks):
            if i not in [5] or addmem is False:
                x = blk(x, H_t, W_t, H_s, W_s)
            else:
                x, mem, z = blk.forward_mem(x, H_t, W_t, H_s, W_s, mem, z)

                # new_mems.append(new_mem.detach())
                # new_zs.append(new_z.detach())
                # new_mems.append(new_mem.detach())
                # new_zs.append(new_z.detach())

            if i > 4 and i % 2 == 1 and i < 11:
                mid_x_t, _, mid_x_s, mid_reg = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s, 4], dim=1)
                mid_x_s_2d = mid_x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
                early_reg_tokens.append(mid_reg)
                early_search_tokens.append(mid_x_s_2d)
            distill_feat_list.append(x)

        x_t, x_ot, x_s, reg_tokens = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s, 4], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d, reg_tokens, distill_feat_list, early_search_tokens, early_reg_tokens, mem, z

    def forward_test(self, x_t, x_ot, x_s, mid_score_head):
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.feat_sz_s
        H_t = W_t = self.feat_sz_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        #
        x = torch.cat([x_t, x_ot, x_s], dim=1)  # (b, hw+hw+HW+4, embed_dim)
        x = self.pos_drop(x)
        flag = 0

        for i, blk in enumerate(self.blocks):
            x = blk.forward_test(x, H_t, W_t, H_s, W_s)
            flag = flag + 1
            if i % 2 == 1 and i > 4 and i < 11:
                #            if i>2 and i<6:
                #                print(0)
                mid_x = self.mid_blocks[i // 2 - 2](x, H_t, W_t, H_s, W_s)  # 调制层
                mid_x_t, mid_x_ot, mid_x_s = torch.split(mid_x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)
                x_s_2d = mid_x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
                score = mid_score_head(x_s_2d, mid_x_t).view(-1)
                # print(score)
                # if score<0.7:
                #    print(score)
                if score > 0.8:
                    x = mid_x
                    break
        x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)
        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        # x = torch.split(x, [H_s * W_s], dim=1)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return x_t_2d, x_ot_2d, x_s_2d, flag

    def set_online(self, x_t, x_ot):
        x_t = self.patch_embed(x_t)
        x_ot = self.patch_embed(x_ot)

        H_t = W_t = self.feat_sz_t

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_ot = x_ot.reshape(1, -1, x_ot.size(-1))  # [1, num_ot * H_t * W_t, C]
        x = torch.cat([x_t, x_ot], dim=1)

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk.set_online(x, H_t, W_t)

        x_t = x[:, :H_t * W_t]
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=H_t, w=W_t)

        self.template = x_t


def get_mixformer_vit(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    if config.MODEL.VIT_TYPE == 'large_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, reg_tokens_cfg=config.MODEL.REG_TOKENS)
    elif config.MODEL.VIT_TYPE == 'base_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=768, depth=config.MODEL.BACKBONE.DEPTH, num_heads=12,
            mlp_ratio=config.MODEL.BACKBONE.MLP_RATIO, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, reg_tokens_cfg=config.MODEL.REG_TOKENS)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")

    return vit


class MixFormer(nn.Module):
    """ Mixformer tracking with mem tokens, whcih jointly perform feature extraction and interaction. """

    def __init__(self, backbone, box_head, early_box_head_6, early_box_head_8, early_box_head_10, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.num_heads = 12
        self.dim = 64
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type
        self.early_box_head_6 = early_box_head_6
        self.early_box_head_8 = early_box_head_8
        self.early_box_head_10 = early_box_head_10
        self.mem1 = nn.Parameter(torch.randn(1, self.num_heads, self.dim, self.dim))
        # self.mem2 = nn.Parameter(torch.randn(1, self.num_heads, self.dim, self.dim))
        # self.mem3 = nn.Parameter(torch.randn(1, self.num_heads, self.dim, self.dim))
        # self.mem4 = nn.Parameter(torch.randn(1, self.num_heads, self.dim, self.dim))
        # self.mems = [self.mem1, self.mem2, self.mem3, self.mem4]
        self.z1 = nn.Parameter(torch.ones(1, self.num_heads, self.dim, 1))
        # self.z2 = nn.Parameter(torch.ones(1, self.num_heads, self.dim, 1))
        # self.z3 = nn.Parameter(torch.ones(1, self.num_heads, self.dim, 1))
        # self.z4 = nn.Parameter(torch.ones(1, self.num_heads, self.dim, 1))
        # self.zs = [self.z1, self.z2, self.z3, self.z4]

    def forward(self, template, online_template, search, softmax, run_score_head=True, gt_bboxes=None, addmem=True):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        all_output = []
        all_boxes = []
        if addmem == True:
            mem = None
            z = None

        for i in range(len(search)):
            if addmem is True:
                if mem is None:
                    f_template, _, f_search, reg_tokens, distill_feat_list, early_search_tokens, early_reg_tokens, mem, z = self.backbone(
                        template, online_template, search[i], self.mem1, self.z1, addmem)
                    # mem = mem.detach()
                    # z = z.detach()
                else:
                    f_template, _, f_search, reg_tokens, distill_feat_list, early_search_tokens, early_reg_tokens, mem, z = self.backbone(
                        template, online_template, search[i], mem, z, addmem)
                    # mem = mem.detach()
                    # z = z.detach()
            else:
                f_template, _, f_search, reg_tokens, distill_feat_list, early_search_tokens, early_reg_tokens, mem, z = self.backbone(
                    template, online_template, search[i], self.mem1, self.z1, addmem)

            out = self.forward_head(f_template, f_search, reg_tokens=reg_tokens, softmax=softmax)
            early_box_6 = self.forward_box_head_6(f_template, early_search_tokens[0], reg_tokens=early_reg_tokens[0],
                                                  softmax=softmax)
            early_box_8 = self.forward_box_head_8(f_template, early_search_tokens[1], reg_tokens=early_reg_tokens[1],
                                                  softmax=softmax)
            early_box_10 = self.forward_box_head_10(f_template, early_search_tokens[2], reg_tokens=early_reg_tokens[2],
                                                    softmax=softmax)
            early_boxes = [early_box_6, early_box_8, early_box_10]

            out['reg_tokens'] = reg_tokens
            out['distill_feat_list'] = distill_feat_list
            all_output.append(out)
            all_boxes.append(early_boxes)
        if addmem is True:
            del mem, z
        return all_output, all_boxes

    def forward_test(self, template, online_template, search, softmax, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)

        if template.dim() == 5:
            template = template.squeeze(0)
        # if online_template.dim() == 5:
        #    online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search, flag = self.backbone.forward_test(template, online_template, search,
                                                                             self.mid_score_head)
        # Forward the corner head and score head
        out_dict = {}
        if flag == 4:
            out_dict_box = self.forward_box_head_6(template, search, softmax=softmax)
            out_dict.update(out_dict_box)

        elif flag == 5:
            out_dict_box = self.forward_box_head_8(template, search, softmax=softmax)
            out_dict.update(out_dict_box)

        elif flag == 6:
            out_dict_box = self.forward_box_head_10(template, search, softmax=softmax)
            out_dict.update(out_dict_box)

        else:
            out_dict = self.forward_head(search, template, run_score_head=run_score_head, softmax=softmax)

        return out_dict, None

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_head(self, template, search, softmax, reg_tokens=None):
        """
        :param search: (b, c, h, w), reg_mask: (b, h, w)
        :return:
        """
        out_dict = {}
        out_dict_box = self.forward_box_head(template, search, reg_tokens=reg_tokens, softmax=softmax)
        out_dict.update(out_dict_box)

        return out_dict

    def forward_box_head(self, template, search, softmax, reg_tokens=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "MLP" == self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_l, prob_t, prob_r, prob_b = self.box_head(reg_tokens, softmax=softmax)
            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_l': prob_l,
                'prob_t': prob_t,
                'prob_b': prob_b,
                'prob_r': prob_r,
            }
            return out
        elif "CS_head" in self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_c = self.box_head(search, reg_tokens, softmax=softmax)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
            }
            return out
        elif "MLP_head2" == self.head_type:
            b = search.size(0)
            pred_boxes, prob_c, prob_reg = self.box_head(template, search)
            # plt.matshow(prob_c.reshape(18,18).cpu().numpy())
            # plt.show()
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        else:
            raise KeyError

    def forward_box_head_4(self, template, search, softmax, reg_tokens=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "MLP" == self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_l, prob_t, prob_r, prob_b = self.early_box_head_6(reg_tokens, softmax=softmax)
            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_l': prob_l,
                'prob_t': prob_t,
                'prob_b': prob_b,
                'prob_r': prob_r,
            }
            return out
        elif "CS_head" in self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_c = self.box_head(search, reg_tokens, softmax=softmax)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
            }
            return out
        elif "MLP_head2" == self.head_type:
            b = search.size(0)
            pred_boxes, prob_c, prob_reg = self.early_box_head_6(template, search)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        else:
            raise KeyError

    def forward_box_head_6(self, template, search, softmax, reg_tokens=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "MLP" == self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_l, prob_t, prob_r, prob_b = self.early_box_head_6(reg_tokens, softmax=softmax)
            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_l': prob_l,
                'prob_t': prob_t,
                'prob_b': prob_b,
                'prob_r': prob_r,
            }
            return out
        elif "CS_head" in self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_c = self.box_head(search, reg_tokens, softmax=softmax)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
            }
            return out
        elif "MLP_head2" == self.head_type:
            b = search.size(0)
            pred_boxes, prob_c, prob_reg = self.early_box_head_6(template, search)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        else:
            raise KeyError

    def forward_box_head_8(self, template, search, softmax, reg_tokens=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "MLP" == self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_l, prob_t, prob_r, prob_b = self.early_box_head_8(reg_tokens, softmax=softmax)
            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_l': prob_l,
                'prob_t': prob_t,
                'prob_b': prob_b,
                'prob_r': prob_r,
            }
            return out
        elif "CS_head" in self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_c = self.box_head(search, reg_tokens, softmax=softmax)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
            }
            return out
        elif "MLP_head2" == self.head_type:
            b = search.size(0)
            pred_boxes, prob_c, prob_reg = self.early_box_head_8(template, search)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        else:
            raise KeyError

    def forward_box_head_10(self, template, search, softmax, reg_tokens=None):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "MLP" == self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_l, prob_t, prob_r, prob_b = self.early_box_head_10(reg_tokens, softmax=softmax)
            outputs_coord = box_xyxy_to_cxcywh(pred_boxes)
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_l': prob_l,
                'prob_t': prob_t,
                'prob_b': prob_b,
                'prob_r': prob_r,
            }
            return out
        elif "CS_head" in self.head_type:
            b = reg_tokens.size(0)
            pred_boxes, prob_c = self.box_head(search, reg_tokens, softmax=softmax)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
            }
            return out
        elif "MLP_head2" == self.head_type:
            b = search.size(0)
            pred_boxes, prob_c, prob_reg = self.early_box_head_10(template, search)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            #            print(1)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        else:
            raise KeyError


def build_mixformer_vit_mem(cfg, settings=None, train=True) -> MixFormer:
    backbone = get_mixformer_vit(cfg, train)  # backbone without positional encoding and attention mask
    box_head = build_box_head(cfg)  # a simple corner head
    early_box_head_6 = build_box_head(cfg)
    early_box_head_8 = build_box_head(cfg)
    early_box_head_10 = build_box_head(cfg)
    model = MixFormer(
        backbone,
        box_head,
        early_box_head_6,
        early_box_head_8,
        early_box_head_10,
        head_type=cfg.MODEL.HEAD_TYPE
    )
    init_early_box_head = False

    if cfg.MODEL.PRETRAINED_STAGE1 and train:
        ckpt_path = settings.stage1_model
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_net = ckpt['net']
        if init_early_box_head:
            ckpt_net["early_box_head_6.layers_l.0.0.weight"] = ckpt_net["box_head.layers_l.0.0.weight"]
            ckpt_net["early_box_head_6.layers_l.0.0.bias"] = ckpt_net["box_head.layers_l.0.0.bias"]
            ckpt_net["early_box_head_6.layers_l.0.1.weight"] = ckpt_net["box_head.layers_l.0.1.weight"]
            ckpt_net["early_box_head_6.layers_l.0.1.bias"] = ckpt_net["box_head.layers_l.0.1.bias"]
            ckpt_net["early_box_head_6.layers_l.1.0.weight"] = ckpt_net["box_head.layers_l.1.0.weight"]
            ckpt_net["early_box_head_6.layers_l.1.0.bias"] = ckpt_net["box_head.layers_l.1.0.bias"]
            ckpt_net["early_box_head_6.layers_l.1.1.weight"] = ckpt_net["box_head.layers_l.1.1.weight"]
            ckpt_net["early_box_head_6.layers_l.1.1.bias"] = ckpt_net["box_head.layers_l.1.1.bias"]

            ckpt_net["early_box_head_8.layers_l.0.0.weight"] = ckpt_net["box_head.layers_l.0.0.weight"]
            ckpt_net["early_box_head_8.layers_l.0.0.bias"] = ckpt_net["box_head.layers_l.0.0.bias"]
            ckpt_net["early_box_head_8.layers_l.0.1.weight"] = ckpt_net["box_head.layers_l.0.1.weight"]
            ckpt_net["early_box_head_8.layers_l.0.1.bias"] = ckpt_net["box_head.layers_l.0.1.bias"]
            ckpt_net["early_box_head_8.layers_l.1.0.weight"] = ckpt_net["box_head.layers_l.1.0.weight"]
            ckpt_net["early_box_head_8.layers_l.1.0.bias"] = ckpt_net["box_head.layers_l.1.0.bias"]
            ckpt_net["early_box_head_8.layers_l.1.1.weight"] = ckpt_net["box_head.layers_l.1.1.weight"]
            ckpt_net["early_box_head_8.layers_l.1.1.bias"] = ckpt_net["box_head.layers_l.1.1.bias"]

            ckpt_net["early_box_head_10.layers_l.0.0.weight"] = ckpt_net["box_head.layers_l.0.0.weight"]
            ckpt_net["early_box_head_10.layers_l.0.0.bias"] = ckpt_net["box_head.layers_l.0.0.bias"]
            ckpt_net["early_box_head_10.layers_l.0.1.weight"] = ckpt_net["box_head.layers_l.0.1.weight"]
            ckpt_net["early_box_head_10.layers_l.0.1.bias"] = ckpt_net["box_head.layers_l.0.1.bias"]
            ckpt_net["early_box_head_10.layers_l.1.0.weight"] = ckpt_net["box_head.layers_l.1.0.weight"]
            ckpt_net["early_box_head_10.layers_l.1.0.bias"] = ckpt_net["box_head.layers_l.1.0.bias"]
            ckpt_net["early_box_head_10.layers_l.1.1.weight"] = ckpt_net["box_head.layers_l.1.1.weight"]
            ckpt_net["early_box_head_10.layers_l.1.1.bias"] = ckpt_net["box_head.layers_l.1.1.bias"]

        missing_keys, unexpected_keys = model.load_state_dict(ckpt['net'], strict=False)
        if is_main_process():
            print("Loading pretrained mixformer weights from {}.".format(ckpt_path))
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained mixformer weights done.")

    return model
