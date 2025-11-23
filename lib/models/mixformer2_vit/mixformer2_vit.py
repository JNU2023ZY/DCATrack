import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp, trunc_normal_
from torch.nn.init import xavier_uniform_
from lib.utils.misc import is_main_process
from lib.models.mixformer2_vit.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.mixformer_vit.pos_util import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_2

from einops import rearrange
from itertools import repeat
import collections.abc
from lib.models.mixformer2_vit.functions import get_attn_mask
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

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w + 4], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w + 4], dim=2)

        # asymmetric mixed attention
        attn_mt = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn_mt.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h * t_w * 2, C)

        #plt.imshow(attn_mask[0][0][25][128:128+324].reshape(18,18).cpu().detach().numpy())
        #plt.show()

        attn_s = (q_s @ k.transpose(-2, -1)) * self.scale #10,12,326,454
        #attn.masked_fill_(attn_mask, float('-inf')) #local attention mask

        attn = attn_s.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w + 4, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        attn_s = (q_s @ q_s.transpose(-2, -1)) * self.scale
        
        q_s_out, _ = torch.split(q_s, [s_h * s_w, 4],dim=2)
        q_rel = (q_s_out @ q_s_out.transpose(-2, -1)) * self.scale
        print("q_rel")
        
        return x, attn_mt, q_rel.softmax(dim=-1), q_rel.softmax(dim=-1), k, v      
#        return x, attn_mt, attn_s.softmax(dim=-1), q_rel.softmax(dim=-1), k, v

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
        #attn.masked_fill_(attn_mask, float('-inf'))  # local attention mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w + 4, C)

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

    def forward(self, x, t_h, t_w, s_h, s_w,remove_rate=1.0):
        xl, mt_map,s_map, q, k, v = self.attn(self.norm1(x), t_h, t_w, s_h, s_w)
        x = x + remove_rate * self.drop_path1(xl)
        x = x + remove_rate * self.drop_path2(self.mlp(self.norm2(x)))
        return x, mt_map, s_map, q, k, v

    def forward_test(self, x, s_h, s_w):
        x = x + self.drop_path1(self.attn.forward_test(self.norm1(x), s_h, s_w))
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
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None, bsz=None):
        super(timm.models.vision_transformer.VisionTransformer, self).__init__()

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])
#        self.mid_blocks = nn.Sequential(*[
#            Block(
#                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
#                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
#                norm_layer=norm_layer) for i in range((depth-6)//2)]) #去掉前两层
        self.feat_sz_s = img_size_s // patch_size
        self.feat_sz_t = img_size_t // patch_size
        self.num_patches_s = self.feat_sz_s ** 2
        self.num_patches_t = self.feat_sz_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        #self.reg_tokens = torch.zeros([1, 4, embed_dim],dtype=torch.float)
        self.reg_tokens = nn.Parameter(torch.randn(1, 4, embed_dim))
        self.pos_embed_reg = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.init_pos_embed()
        #self.reg_tokens_cfg = reg_tokens_cfg
        #trunc_normal_(self.reg_tokens, std=.02)
        #self.attn_mask_2 = get_attn_mask(bsz, 12, reg_token_num=2)
        #self.attn_mask_4 = get_attn_mask(bsz, 12, reg_token_num=4)

    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                            cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

        #pos_embed_h, pos_embed_w = get_2d_sincos_pos_embed_2(self.pos_embed_s.shape[-1]*2, int(self.num_patches_s ** .5),
        #                                      cls_token=False)
        #self.reg_tokens[0][0] = torch.from_numpy(np.sum(pos_embed_w, axis=0) / pos_embed_w.shape[0])
        #self.reg_tokens[0][1] = torch.from_numpy(np.sum(pos_embed_h,axis=0)/pos_embed_h.shape[0])
        #self.reg_tokens[0][2] = torch.from_numpy(pos_embed_h[13]-pos_embed_h[5])
        #self.reg_tokens[0][3] = torch.from_numpy(pos_embed_h[13]-pos_embed_h[5])
        #self.reg_tokens = nn.Parameter(self.reg_tokens.cuda().requires_grad_())
        if is_main_process():
            print("Initialize pos embed with fixed sincos embedding.")

    def forward(self, x_t, x_ot, x_s, remove_rate=1.0):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1) #10 768
        H_s = W_s = self.feat_sz_s #18
        H_t = W_t = self.feat_sz_t #8

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        
        reg_tokens = self.reg_tokens.expand(B, -1, -1)  # (b, 4, embed_dim)
        reg_tokens = reg_tokens + self.pos_embed_reg

        #reg_tokens = self.reg_tokens[:,:2,:].expand(B, -1, -1)  # (b, 4, embed_dim)
        #reg_tokens = reg_tokens + self.pos_embed_reg[:,:2,:]
        
        x = torch.cat([x_t, x_ot, x_s, reg_tokens], dim=1)  # (b, hw+hw+HW+4, embed_dim)
        x = self.pos_drop(x)

        distill_feat_list = []
        distill_mt_map_list = []
        distill_s_map_list = []
        distill_q_list = []
        distill_k_list = []
        distill_v_list = []
        early_search_tokens = []
        early_reg_tokens = []
        for i, blk in enumerate(self.blocks):
            #if i in [3, 6, 8, 10]:
            if i in [4]:
#            if i in [12]:
                x, mt_map, s_map, q, k, v = blk(x, H_t, W_t, H_s, W_s, remove_rate)
            else:
                x, mt_map, s_map, q, k, v = blk(x, H_t, W_t, H_s, W_s)
#            if i>0 and i%2==1 and i<11:
            if i < 4:
#            if i < 7 and i > 3:
# #                mid_x = self.mid_blocks[(i//2)-2](x, H_t, W_t, H_s, W_s) #调制层
                _, _, mid_x_s, mid_reg = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s, 4], dim=1)
                mid_x_s_2d = mid_x_s.transpose(1, 2).reshape(B, C, H_s, W_s)
                early_search_tokens.append(mid_x_s_2d)
                early_reg_tokens.append(mid_reg)
            distill_feat_list.append(x)
            distill_mt_map_list.append(mt_map)
            distill_q_list.append(q)
            distill_k_list.append(k)
            distill_v_list.append(v)
            distill_s_map_list.append(s_map)

        x_t, x_ot, x_s, reg_tokens = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s, 4], dim=1)

        #x_s = early_search_tokens[0]  # 使用浅层特征预测2

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d, reg_tokens, distill_feat_list, distill_mt_map_list, distill_s_map_list, distill_q_list, distill_k_list, distill_v_list, early_search_tokens, early_reg_tokens

    def forward_test(self, x):
        x = self.patch_embed(x)
        H_s = W_s = self.feat_sz_s

        x = x + self.pos_embed_s
        reg_tokens = self.reg_tokens.expand(x.size(0), -1, -1)
        reg_tokens = reg_tokens + self.pos_embed_reg
        x = torch.cat([x, reg_tokens], dim=1)
        x = self.pos_drop(x)

        #attn_mask = self.attn_mask_2
        for i, blk in enumerate(self.blocks):
            #if i==6:
            #    reg_tokens = self.reg_tokens[:,2:4,:].expand(B, -1, -1)+self.pos_embed_reg[:,2:4,:]
            #    x = torch.cat([x,reg_tokens], dim=1)
            #    attn_mask = self.attn_mask_4
            x = blk.forward_test(x, H_s, W_s)

        x, reg_tokens = torch.split(x, [H_s * W_s, self.reg_tokens_cfg[-1]], dim=1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return self.template, x, reg_tokens

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
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, bsz=config.TRAIN.BATCH_SIZE)
    elif config.MODEL.VIT_TYPE == 'base_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=768, depth=config.MODEL.BACKBONE.DEPTH, num_heads=12, mlp_ratio=config.MODEL.BACKBONE.MLP_RATIO, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, bsz=config.TRAIN.BATCH_SIZE)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")


    return vit

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=768, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, reg_tokens,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # self-attention
        #q = k = self.with_pos_embed(tgt, query_pos)  # Add object query to the query and key
        #if self.divide_norm:
        #    q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
        #    k = k / torch.norm(k, dim=-1, keepdim=True)
        #tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
        #                      key_padding_mask=tgt_key_padding_mask)[0]
        #tgt = tgt + self.dropout1(tgt2)
        #tgt = self.norm1(tgt)

        bs, cs, Hs, Ws = memory.shape
        tgt = tgt.repeat(bs,1,1)
        mix_memory = torch.cat([tgt, reg_tokens, memory.reshape(bs, cs, Hs*Ws).permute(0,2,1)], dim=1)#[bs,1+2+324,768]
        mix_memory = mix_memory.permute(1,0,2)
        mix_memory2 = self.self_attn(mix_memory,mix_memory,mix_memory)[0]
        #mix_memory = mix_memory + self.dropout1(mix_memory2)
        mix_memory = mix_memory + mix_memory2
        #mix_memory = self.norm1(mix_memory)

        tgt = mix_memory[0:1,:,:]
        reg_tokens = mix_memory[1:3,:,:]

        # mutual attention
        reg_tokens2 = self.multihead_attn(tgt, reg_tokens, reg_tokens)[0]

        #reg_tokens = reg_tokens + self.dropout2(reg_tokens2)
        reg_tokens = reg_tokens + reg_tokens2
        #reg_tokens = self.norm2(reg_tokens)
        #reg_tokens2 = self.linear2(self.dropout(self.activation(self.linear1(reg_tokens))))
        reg_tokens2 = self.linear2(self.activation(self.linear1(reg_tokens)))
        #reg_tokens = reg_tokens + self.dropout3(reg_tokens2)
        reg_tokens = reg_tokens + reg_tokens2
        reg_tokens = self.norm3(reg_tokens)
        return reg_tokens

class MixFormer(nn.Module):
    """ Mixformer tracking with score prediction module, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head, early_box_head_2, early_box_head_4, early_box_head_6, early_box_head_8,early_box_head_10, head_type="CORNER", d_model=768, nhead=8, dim_feedforward=768, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.early_box_head_2 = early_box_head_2
        self.early_box_head_4 = early_box_head_4
        self.early_box_head_6 = early_box_head_6
        self.early_box_head_8 = early_box_head_8
        self.early_box_head_10 = early_box_head_10
        self.head_type = head_type

    def forward(self, template, online_template, search, softmax, remove_rate=1.0, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search, reg_tokens, distill_feat_list, distill_mt_map_list, distill_s_map_list, distill_q_list,distill_k_list, distill_v_list, early_search_tokens, early_reg_tokens = self.backbone(template, online_template, search, remove_rate)
#        template, online_template, search, reg_tokens, distill_feat_list = self.backbone(template, online_template, search)
        #centre_tokens = self.centre_decoder(self.centre_q, search, reg_tokens[:,:2,:])
        #scale_tokens = self.scale_decoder(self.scale_q, search, reg_tokens[:,2:4,:])
        #reg_tokens = torch.cat([centre_tokens,scale_tokens],dim=0).permute(1,0,2)
        # Forward the corner head and score head
        out = self.forward_head(template, search, reg_tokens=reg_tokens, softmax=softmax)
#        print(len(early_search_tokens))
#        early_boxes = [self.forward_box_head(template, early_search_tokens[i], softmax=softmax) for i in
#                       range(len(early_search_tokens))]
#        early_box_2 = self.forward_box_head_6(template, early_search_tokens[0], reg_tokens= early_reg_tokens[0], softmax=softmax)
#        early_box_4 = self.forward_box_head_6(template, early_search_tokens[1], reg_tokens= early_reg_tokens[1], softmax=softmax)
        early_box_6 = self.forward_box_head_6(template, early_search_tokens[0], reg_tokens= early_reg_tokens[0], softmax=softmax)
        early_box_8 = self.forward_box_head_8(template, early_search_tokens[1], reg_tokens= early_reg_tokens[1], softmax=softmax)
        early_box_10 = self.forward_box_head_10(template, early_search_tokens[2], reg_tokens= early_reg_tokens[2], softmax=softmax)
#        early_boxes = [early_box_2,early_box_4,early_box_6, early_box_8, early_box_10]
        early_boxes = [early_box_6, early_box_8, early_box_10]
        #out=early_boxes[4] #使用浅层特征预测4
        out['reg_tokens'] = reg_tokens
        out['distill_feat_list'] = distill_feat_list
        out['distill_mt_map_list'] = distill_mt_map_list
        out['distill_s_map_list'] = distill_s_map_list
        out['distill_q_list'] = distill_q_list
        out['distill_k_list'] = distill_k_list
        out['distill_v_list'] = distill_v_list

        return out, early_boxes, None
        # return out, early_boxes, None
        # return out

    def forward_test(self, search, softmax, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone.forward_test(search)

        # Forward the corner head and score head
        out, outputs_coord_new = self.forward_head(template, search, softmax=softmax)

        return out, outputs_coord_new

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
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        elif "CORNER_UP" == self.head_type:
            b = search.size(0)
            pred_boxes, prob_vec_tl, prob_vec_br = self.box_head(template, search, softmax=softmax)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_tl': prob_vec_tl,
                'prob_br': prob_vec_br, # (b, h*w)
            }
            return out
        else:
            raise KeyError
            
    def forward_box_head_2(self, template, search, softmax, reg_tokens=None):
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
            pred_boxes, prob_c, prob_reg  = self.early_box_head_6(template, search)
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
            pred_boxes, prob_c, prob_reg  = self.early_box_head_6(template, search)
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
            pred_boxes, prob_c,prob_reg = self.early_box_head_10(template, search)
            outputs_coord_new = pred_boxes.view(b, 1, 4)
            out = {
                'pred_boxes': outputs_coord_new,
                'prob_center': prob_c,
                'prob_reg': prob_reg,
            }
            return out
        else:
            raise KeyError

def build_mixformer_vit(cfg, train=False, teacher=False) -> MixFormer:
    backbone = get_mixformer_vit(cfg, train)  # backbone without positional encoding and attention mask
    box_head = build_box_head(cfg)  # a simple corner head
    early_box_head_2 = build_box_head(cfg)
    early_box_head_4 = build_box_head(cfg)
    early_box_head_6 = build_box_head(cfg)
    early_box_head_8 = build_box_head(cfg)
    early_box_head_10 = build_box_head(cfg)
#    early_box_heads = [early_box_head_6, early_box_head_8, early_box_head_10]
    model = MixFormer(
        backbone,
        box_head,
        early_box_head_2,
        early_box_head_4,
        early_box_head_6,
        early_box_head_8,
        early_box_head_10,
        head_type=cfg.MODEL.HEAD_TYPE
    )
    if cfg.MODEL.BACKBONE.PRETRAINED and train:
        print("load student ckpt")
        ckpt_path = cfg.MODEL.BACKBONE.PRETRAINED_PATH
        print(ckpt_path)
        #ckpt_net = torch.load(ckpt_path, map_location='cpu',weights_only=False)['model']
        ckpt_net = torch.load(ckpt_path, map_location='cpu',weights_only=False)['net']
        new_dict = {}
        # ckpt_net["early_box_head_6.layers.0.0.weight"] = ckpt_net["box_head.layers.0.0.weight"]
        # ckpt_net["early_box_head_6.layers.0.0.bias"] = ckpt_net["box_head.layers.0.0.bias"]
        # ckpt_net["early_box_head_6.layers.0.1.weight"] = ckpt_net["box_head.layers.0.1.weight"]
        # ckpt_net["early_box_head_6.layers.0.1.bias"] = ckpt_net["box_head.layers.0.1.bias"]
        # ckpt_net["early_box_head_6.layers.1.0.weight"] = ckpt_net["box_head.layers.1.0.weight"]
        # ckpt_net["early_box_head_6.layers.1.0.bias"] = ckpt_net["box_head.layers.1.0.bias"]
        # ckpt_net["early_box_head_6.layers.1.1.weight"] = ckpt_net["box_head.layers.1.1.weight"]
        # ckpt_net["early_box_head_6.layers.1.1.bias"] = ckpt_net["box_head.layers.1.1.bias"]
        # 
        # ckpt_net["early_box_head_8.layers.0.0.weight"] = ckpt_net["box_head.layers.0.0.weight"]
        # ckpt_net["early_box_head_8.layers.0.0.bias"] = ckpt_net["box_head.layers.0.0.bias"]
        # ckpt_net["early_box_head_8.layers.0.1.weight"] = ckpt_net["box_head.layers.0.1.weight"]
        # ckpt_net["early_box_head_8.layers.0.1.bias"] = ckpt_net["box_head.layers.0.1.bias"]
        # ckpt_net["early_box_head_8.layers.1.0.weight"] = ckpt_net["box_head.layers.1.0.weight"]
        # ckpt_net["early_box_head_8.layers.1.0.bias"] = ckpt_net["box_head.layers.1.0.bias"]
        # ckpt_net["early_box_head_8.layers.1.1.weight"] = ckpt_net["box_head.layers.1.1.weight"]
        # ckpt_net["early_box_head_8.layers.1.1.bias"] = ckpt_net["box_head.layers.1.1.bias"]
        # 
        # ckpt_net["early_box_head_10.layers.0.0.weight"] = ckpt_net["box_head.layers.0.0.weight"]
        # ckpt_net["early_box_head_10.layers.0.0.bias"] = ckpt_net["box_head.layers.0.0.bias"]
        # ckpt_net["early_box_head_10.layers.0.1.weight"] = ckpt_net["box_head.layers.0.1.weight"]
        # ckpt_net["early_box_head_10.layers.0.1.bias"] = ckpt_net["box_head.layers.0.1.bias"]
        # ckpt_net["early_box_head_10.layers.1.0.weight"] = ckpt_net["box_head.layers.1.0.weight"]
        # ckpt_net["early_box_head_10.layers.1.0.bias"] = ckpt_net["box_head.layers.1.0.bias"]
        # ckpt_net["early_box_head_10.layers.1.1.weight"] = ckpt_net["box_head.layers.1.1.weight"]
        # ckpt_net["early_box_head_10.layers.1.1.bias"] = ckpt_net["box_head.layers.1.1.bias"]





#        ckpt_net["early_box_head_6.layers_l.0.0.weight"] = ckpt_net["box_head.layers_l.0.0.weight"]
#        ckpt_net["early_box_head_6.layers_l.0.0.bias"] = ckpt_net["box_head.layers_l.0.0.bias"]
#        ckpt_net["early_box_head_6.layers_l.0.1.weight"] = ckpt_net["box_head.layers_l.0.1.weight"]
#        ckpt_net["early_box_head_6.layers_l.0.1.bias"] = ckpt_net["box_head.layers_l.0.1.bias"]
#        ckpt_net["early_box_head_6.layers_l.1.0.weight"] = ckpt_net["box_head.layers_l.1.0.weight"]
#        ckpt_net["early_box_head_6.layers_l.1.0.bias"] = ckpt_net["box_head.layers_l.1.0.bias"]
#        ckpt_net["early_box_head_6.layers_l.1.1.weight"] = ckpt_net["box_head.layers_l.1.1.weight"]
#        ckpt_net["early_box_head_6.layers_l.1.1.bias"] = ckpt_net["box_head.layers_l.1.1.bias"]

#        ckpt_net["early_box_head_8.layers_l.0.0.weight"] = ckpt_net["box_head.layers_l.0.0.weight"]
#        ckpt_net["early_box_head_8.layers_l.0.0.bias"] = ckpt_net["box_head.layers_l.0.0.bias"]
#        ckpt_net["early_box_head_8.layers_l.0.1.weight"] = ckpt_net["box_head.layers_l.0.1.weight"]
#        ckpt_net["early_box_head_8.layers_l.0.1.bias"] = ckpt_net["box_head.layers_l.0.1.bias"]
#        ckpt_net["early_box_head_8.layers_l.1.0.weight"] = ckpt_net["box_head.layers_l.1.0.weight"]
#        ckpt_net["early_box_head_8.layers_l.1.0.bias"] = ckpt_net["box_head.layers_l.1.0.bias"]
#        ckpt_net["early_box_head_8.layers_l.1.1.weight"] = ckpt_net["box_head.layers_l.1.1.weight"]
#        ckpt_net["early_box_head_8.layers_l.1.1.bias"] = ckpt_net["box_head.layers_l.1.1.bias"]

#        ckpt_net["early_box_head_10.layers_l.0.0.weight"] = ckpt_net["box_head.layers_l.0.0.weight"]
#        ckpt_net["early_box_head_10.layers_l.0.0.bias"] = ckpt_net["box_head.layers_l.0.0.bias"]
#        ckpt_net["early_box_head_10.layers_l.0.1.weight"] = ckpt_net["box_head.layers_l.0.1.weight"]
#        ckpt_net["early_box_head_10.layers_l.0.1.bias"] = ckpt_net["box_head.layers_l.0.1.bias"]
#        ckpt_net["early_box_head_10.layers_l.1.0.weight"] = ckpt_net["box_head.layers_l.1.0.weight"]
#        ckpt_net["early_box_head_10.layers_l.1.0.bias"] = ckpt_net["box_head.layers_l.1.0.bias"]
#        ckpt_net["early_box_head_10.layers_l.1.1.weight"] = ckpt_net["box_head.layers_l.1.1.weight"]
#        ckpt_net["early_box_head_10.layers_l.1.1.bias"] = ckpt_net["box_head.layers_l.1.1.bias"]
        for k, v in ckpt_net.items():
            #if 'pos_embed' not in k and 'mask_token' not in k:
            new_dict[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained model from {}\n".format(ckpt_path))
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")

    return model
