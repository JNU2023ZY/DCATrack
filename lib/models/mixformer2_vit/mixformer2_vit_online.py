import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp, trunc_normal_
from torch.nn.init import xavier_uniform_
from lib.utils.misc import is_main_process
from lib.models.mixformer2_vit.head import build_box_head, build_score_decoder, build_score_decoder2,build_mid_score_decoder
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w ], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w ], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w ], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w * 2, C)

        #plt.imshow(attn_mask[0][0][25][128:128+324].reshape(18,18).cpu().detach().numpy())
        #plt.show()

        attn = (q_s @ k.transpose(-2, -1)) * self.scale #10,12,326,454

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_test(self, x, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, _, _ = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        _, k, v = qkv.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_online(self, x, t_h, t_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

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
        self.mid_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range((depth - 6) // 2)]) #去掉前两层
        self.feat_sz_s = img_size_s // patch_size
        self.feat_sz_t = img_size_t // patch_size
        self.num_patches_s = self.feat_sz_s ** 2
        self.num_patches_t = self.feat_sz_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        #self.reg_tokens = nn.Parameter(torch.randn(1, 2, embed_dim))
        #self.pos_embed_reg = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        #self.reg_tokens_cfg = reg_tokens_cfg
        self.init_pos_embed()

        #trunc_normal_(self.reg_tokens, std=.02)

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

    def forward(self, x_t, x_ot, x_s, mid_score_head=None):
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

        #reg_tokens = self.reg_tokens[:, :2, :].expand(B, -1, -1)  # (b, 4, embed_dim)
        #reg_tokens = reg_tokens + self.pos_embed_reg[:, :2, :]

        x = torch.cat([x_t, x_ot, x_s], dim=1)  # (b, hw+hw+HW+4, embed_dim)
        x = self.pos_drop(x)

        distill_feat_list = []
        early_search_tokens = []
        early_scores = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, H_t, W_t, H_s, W_s)
            #if i%2==1 and i>=3 and i<11:
            if i>3 and i % 2 == 1 and i < 11:
                mid_x = self.mid_blocks[(i // 2)-2](x, H_t, W_t, H_s, W_s)  # 调制层
                mid_x_t, _, mid_x_s = torch.split(mid_x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)
                mid_x_s_2d = mid_x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
                #x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)
                #x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
                score = mid_score_head(mid_x_s_2d,mid_x_t).view(-1) #(B,)
                early_scores.append(score)
                early_search_tokens.append(mid_x_s_2d)
            distill_feat_list.append(x)

        x_t, x_ot, x_s = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d, distill_feat_list, early_search_tokens, early_scores

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

        for i, blk in enumerate(self.blocks):
            x = blk.forward_test(x, H_t, W_t, H_s, W_s)
            if i>4 and i%2==1 and i<11:
                mid_x = self.mid_blocks[(i // 2)-2](x, H_t, W_t, H_s, W_s)  # 调制层
                mid_x_t, mid_x_ot, mid_x_s = torch.split(mid_x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)
                x_s_2d = mid_x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
                score = mid_score_head(x_s_2d,mid_x_t).view(-1)
                #print(score)
                #if score<0.7:
                #    print(score)
                if score>0.8:
                    x = mid_x
                    break
        x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)
        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        #x = torch.split(x, [H_s * W_s], dim=1)
        #x = rearrange(x, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return x_t_2d, x_ot_2d, x_s_2d

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
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,reg_tokens_cfg=config.MODEL.REG_TOKENS)
    elif config.MODEL.VIT_TYPE == 'base_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=768, depth=config.MODEL.BACKBONE.DEPTH, num_heads=12, mlp_ratio=config.MODEL.BACKBONE.MLP_RATIO, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,reg_tokens_cfg=config.MODEL.REG_TOKENS)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")

    return vit


class MixFormer(nn.Module):
    """ Mixformer tracking with score prediction module, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head, mid_score_head, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type
        #self.centre_q = nn.Parameter(torch.Tensor(1, 1, 768))
        #xavier_uniform_(self.centre_q)
        #self.scale_q = nn.Parameter(torch.Tensor(1, 1, 768))
        #xavier_uniform_(self.scale_q)
        #self.score_head = score_head
        self.mid_score_head = mid_score_head

    def forward(self, template, online_template, search, softmax, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search, distill_feat_list, early_search_tokens, early_scores= self.backbone(template, online_template, search, self.mid_score_head)
        # Forward the corner head and score head
        out = self.forward_head(search, template, run_score_head=run_score_head, softmax=softmax)
        early_boxes = [self.forward_box_head(template, early_search_tokens[i], softmax=softmax)
                       for i in
                       range(len(early_search_tokens))]
        #out['reg_tokens'] = reg_tokens
        out['distill_feat_list'] = distill_feat_list

        return out, early_boxes, early_scores

    def forward_test(self, template, online_template, search, softmax, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)

        if template.dim() == 5:
            template = template.squeeze(0)
        #if online_template.dim() == 5:
        #    online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search = self.backbone.forward_test(template, online_template, search, self.mid_score_head)
        # Forward the corner head and score head
        out = self.forward_head(search, template, run_score_head=run_score_head, softmax=softmax)

        return out, None

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_head(self, search, template, run_score_head, softmax, gt_bboxes=None):
        """
        :param search: (b, c, h, w), reg_mask: (b, h, w)
        :return:
        """
        out_dict = {}
        out_dict_box = self.forward_box_head(template, search, softmax=softmax)
        out_dict.update(out_dict_box)

        #if run_score_head:
        #    if gt_bboxes is None:
        #        gt_bboxes = box_cxcywh_to_xyxy(out_dict['pred_boxes'].clone().view(-1, 4))
        #    out_dict.update({
        #        #'pred_scores': self.score_head(search, template, gt_bboxes).view(-1),
        #        'pred_scores': self.score_head(search).view(-1),
        #    })
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
            pred_boxes, prob_c = self.box_head(template, search)
            #plt.matshow(prob_c.reshape(18,18).cpu().numpy())
            #plt.show()
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
            }
            return out
        else:
            raise KeyError


def build_mixformer_vit_online(cfg, settings=None, train=True) -> MixFormer:
    backbone = get_mixformer_vit(cfg, train)  # backbone without positional encoding and attention mask
    box_head = build_box_head(cfg)  # a simple corner head
    #score_head = build_score_decoder2(cfg)
    mid_score_head = build_mid_score_decoder(cfg)
    model = MixFormer(
        backbone,
        box_head,
        #score_head,
        mid_score_head,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    if cfg.MODEL.PRETRAINED_STAGE1 and train:
        ckpt_path = settings.stage1_model
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['net'], strict=False)
        if is_main_process():
            print("Loading pretrained mixformer weights from {}.".format(ckpt_path))
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained mixformer weights done.")

    return model