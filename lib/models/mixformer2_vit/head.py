import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.mixformer_vit.utils import FrozenBatchNorm2d
from lib.models.mixformer2_vit.score_decoder import MLPScoreDecoder,MidScoreDecoder
import matplotlib.pyplot as plt
class MlpHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, feat_sz, num_layers, stride, norm=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = feat_sz
        self.img_sz = feat_sz * stride
        if norm:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k), nn.ReLU())
                                          if i < num_layers - 1 
                                          else nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1 
                                          else nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
 
        with torch.no_grad():
            #self.indice = torch.arange(0, feat_sz).unsqueeze(0).half().cuda() * stride # (1, feat_sz)
            self.indice = torch.arange(0, feat_sz).unsqueeze(0).cuda() * stride # (1, feat_sz)
            #self.indice = torch.arange(0, feat_sz).unsqueeze(0).cpu() * stride # (1, feat_sz)

    def forward(self, reg_tokens, softmax):
        """
        reg_tokens shape: (b, 4, embed_dim)
        """
        reg_token_l, reg_token_r, reg_token_t, reg_token_b = reg_tokens.unbind(dim=1)   # (b, c)
        score_l = self.layers(reg_token_l)
        score_r = self.layers(reg_token_r)
        score_t = self.layers(reg_token_t)
        score_b = self.layers(reg_token_b) # (b, feat_sz)

        prob_l = score_l.softmax(dim=-1)
        prob_r = score_r.softmax(dim=-1)
        prob_t = score_t.softmax(dim=-1)
        prob_b = score_b.softmax(dim=-1)
    
        coord_l = torch.sum((self.indice * prob_l), dim=-1)
        coord_r = torch.sum((self.indice * prob_r), dim=-1)
        coord_t = torch.sum((self.indice * prob_t), dim=-1)
        coord_b = torch.sum((self.indice * prob_b), dim=-1) # (b, ) absolute coordinates


        # return xyxy, ltrb
        if softmax:
            return torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
                prob_l, prob_t, prob_r, prob_b
        else:
            return torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
                score_l, score_t, score_r, score_b


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        feat_sz = cfg.MODEL.FEAT_SZ
        stride = cfg.DATA.SEARCH.SIZE / feat_sz
        print("feat size: ", feat_sz, ", stride: ", stride)
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MlpHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            feat_sz=feat_sz,
            num_layers=2,
            stride=stride,
            norm=True
        )
        return mlp_head

    elif "CORNER" in cfg.MODEL.HEAD_TYPE:
        stride = 16
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        # channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
        freeze_bn = getattr(cfg.MODEL, "HEAD_FREEZE_BN", False)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_UP":
            stride = 4
            feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
            corner_head = Corner_Predictor_UP(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        else:
            raise ValueError()
        return corner_head

    elif cfg.MODEL.HEAD_TYPE == "CS_head":
        stride = 16
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        # channel = getattr(cfg.MODEL, "HEAD_DIM", 256)
        channel = getattr(cfg.MODEL, "HEAD_DIM", 384)
        freeze_bn = getattr(cfg.MODEL, "HEAD_FREEZE_BN", False)
        print("head channel: %d" % channel)
        corner_head = CS_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride, freeze_bn=freeze_bn)
        return corner_head
    elif cfg.MODEL.HEAD_TYPE == "MLP_head2":
        stride = 16
        mlp_head = MLP_head2(hidden_dim=cfg.MODEL.HIDDEN_DIM,stride=stride)
        return mlp_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
                # .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
                # .view((self.feat_sz * self.feat_sz,)).float()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

class Corner_Predictor_UP(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor_UP, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.adjust1_tl = conv(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_tl = conv(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_tl = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_tl = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        self.adjust1_br = conv(inplanes, channel // 2, freeze_bn=freeze_bn)
        self.adjust2_br = conv(inplanes, channel // 4, freeze_bn=freeze_bn)

        self.adjust3_br = nn.Sequential(conv(channel // 2, channel // 4, freeze_bn=freeze_bn),
                                        conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))
        self.adjust4_br = nn.Sequential(conv(channel // 4, channel // 8, freeze_bn=freeze_bn),
                                        conv(channel // 8, 1, freeze_bn=freeze_bn))

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        x_init = x
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)

        #up-1
        x_init_up1 = F.interpolate(self.adjust1_tl(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_tl2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_tl3 = self.conv3_tl(x_up1)

        #up-2
        x_init_up2 = F.interpolate(self.adjust2_tl(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_tl3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_tl4 = self.conv4_tl(x_up2)
        score_map_tl = self.conv5_tl(x_tl4) + F.interpolate(self.adjust3_tl(x_tl2), scale_factor=4) + F.interpolate(self.adjust4_tl(x_tl3), scale_factor=2)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)

        # up-1
        x_init_up1 = F.interpolate(self.adjust1_br(x_init), scale_factor=2)
        x_up1 = F.interpolate(x_br2, scale_factor=2)
        x_up1 = x_init_up1 + x_up1

        x_br3 = self.conv3_br(x_up1)

        # up-2
        x_init_up2 = F.interpolate(self.adjust2_br(x_init), scale_factor=4)
        x_up2 = F.interpolate(x_br3, scale_factor=2)
        x_up2 = x_init_up2 + x_up2

        x_br4 = self.conv4_br(x_up2)
        score_map_br = self.conv5_br(x_br4) + F.interpolate(self.adjust3_br(x_br2), scale_factor=4) + F.interpolate(self.adjust4_br(x_br3), scale_factor=2)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

# score
class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg):
    return MlpScoreDecoder(
        in_dim=cfg.MODEL.HIDDEN_DIM,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_layers=2,
        bn=False
    )
def build_score_decoder2(cfg):
    return MLPScoreDecoder(
        num_heads=cfg.MODEL.HIDDEN_DIM // 64,
        hidden_dim=cfg.MODEL.HIDDEN_DIM
        #pool_size=4
    )
def build_mid_score_decoder(cfg):
    return MidScoreDecoder(
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_heads=cfg.MODEL.HIDDEN_DIM//64
    )
    #num_heads=cfg.MODEL.HIDDEN_DIM//64

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_sigmoid(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.Sigmoid())
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.Sigmoid())

class CS_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=18, stride=16, freeze_bn=False):
        super(CS_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_c = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_c = conv(channel, channel // 4, freeze_bn=freeze_bn)
        #self.conv3_c = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        #self.conv4_c = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_c = nn.Conv2d(channel , 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_s = conv_sigmoid(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_s = conv_sigmoid(channel, channel // 4, freeze_bn=freeze_bn)
        #self.conv3_s = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        #self.conv4_s = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        #self.conv5_s = nn.Conv2d(channel // 4, 2, kernel_size=1)
        self.conv5_s = conv_sigmoid(channel // 4, 2, kernel_size=1, padding=0)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
                # .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
                # .view((self.feat_sz * self.feat_sz,)).float()

    def forward(self, x, cs_tokens, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_c, score_map_s = self.get_score_map(x, cs_tokens)
        coorx_c, coory_c, prob_vec_c = self.soft_argmax(score_map_c, return_dist=True, softmax=softmax)
        score_map_s = score_map_s.view(prob_vec_c.size(0),2,prob_vec_c.size(1))
        prob_c = torch.stack([prob_vec_c,prob_vec_c],dim=1)

        #plt.matshow(prob_c[0][0].reshape(18, 18).cpu().detach().numpy())
        #plt.colorbar()
        #plt.show()

        score_map_s = score_map_s * prob_c
        exp_hw = torch.sum(score_map_s,dim=2) * self.feat_sz
        exp_h = exp_hw[:, 0]
        exp_w = exp_hw[:, 1]
        return torch.stack((coorx_c, coory_c, exp_w, exp_h), dim=1) / self.img_sz, prob_vec_c

    def get_score_map(self, x, cs_tokens):
        # x: [bsz,768,18,18]
        # cs_tokens: [bsz,2,768]
        bsz, C, H, W = x.size()
        # center branch
        c_token = cs_tokens[:,0,:].unsqueeze(-1) #[bsz, 768, 1]
        s_token = cs_tokens[:, 1, :].unsqueeze(-1)
        x = x.view(bsz,C,H*W).transpose(1, 2)

        att = torch.matmul(x, c_token) #[bsz,HW,1]
        x_c = (x.unsqueeze(-1) * att.unsqueeze(-2)).transpose(1,2).view(bsz,C,H,W)
        x_c1 = self.conv1_c(x_c)
        #x_c2 = self.conv2_c(x_c1)
        #x_c3 = self.conv3_c(x_c2)
        #x_c4 = self.conv4_c(x_c3)
        score_map_c = self.conv5_c(x_c1)

        # scale branch
        att = torch.matmul(x, s_token)  # [bsz,HW,1]
        x_s = (x.unsqueeze(-1) * att.unsqueeze(-2)).transpose(1, 2).view(bsz, C, H, W)
        x_s1 = self.conv1_s(x_s)
        x_s2 = self.conv2_s(x_s1)
        #x_s3 = self.conv3_s(x_s2)
        #x_s4 = self.conv4_s(x_s3)
        #score_map_s = F.sigmoid(self.conv5_s(x_s2))
        score_map_s = self.conv5_s(x_s2)
        return score_map_c, score_map_s

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_w = nn.Linear(dim, dim, bias=True)
        self.k_w = nn.Linear(dim, dim, bias=True)
        self.v_w = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, t, s):
        """
        x is a concatenated vector of template and search region features.
        """
        B, Ns, C = s.shape
        B, Nt, C = t.shape
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.q_w(s).reshape(B, Ns*self.num_heads, C//self.num_heads)
        k = self.k_w(t).reshape(B, Nt * self.num_heads, C//self.num_heads)
        v = self.v_w(t).reshape(B, Nt * self.num_heads, C//self.num_heads)

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        s = (attn @ v).transpose(1, 2).reshape(B, Ns, C)

        s = self.proj(s)
        s = self.proj_drop(s)
        return s

class MLP_head2(nn.Module):
    def __init__(self, hidden_dim, stride):
        super().__init__()
        self.class_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        hanning = np.hanning(6)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        #self.dec_token = nn.Parameter(torch.zeros(1, 324, hidden_dim))
        #self.multihead_attn = CrossAttention(hidden_dim, 8, attn_drop=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, ht, hs):
        self.window = torch.tensor(self.window,device=hs.device)
        bsz, C, W, H = hs.size()
        hs = hs.view(bsz,C,W*H).transpose(1,2) #8 324 768
        hs = self.norm1(hs)
        # cross attention
        # dec_t = self.dec_token.repeat(bsz,1,1)

        bsz, C, Wt, Ht = ht.size()
        ht = ht.view(bsz, C, Wt * Ht).transpose(1, 2) #8 64 768
        ht = self.norm2(ht)
        #hs = self.multihead_attn(ht, hs)

        query = torch.mean(ht,dim=1,keepdim=True).transpose(1,2)
        attn =torch.matmul(hs,query)
        attn = torch.softmax(attn,dim=1) + 1
        opt =hs*attn

        outputs_class = self.class_embed(opt)
        score = F.softmax(outputs_class, dim=2).data[:, :, 0]
        pscore = score * (1 - 0.49) + self.window * 0.49

        outputs_coord = self.bbox_embed(opt).sigmoid()
        pred_bbox = outputs_coord.transpose(1, 2)
        #score = outputs_class.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)

        #delta = outputs_coord.permute(2, 1, 0).contiguous().view(4, -1)

        best_idx = torch.argmax(pscore, dim=1)
        #bbox = pred_bbox[:, :, best_idx]
        bbox = torch.stack([pred_bbox[i,:,best_idx[i]] for i in range(bsz)]) # l t r b
        #bbox = bbox*288 # 288: search region size
        cx = ((bbox[:,0] + bbox[:,2]) / 2).unsqueeze(-1)
        cy = ((bbox[:,1] + bbox[:,3]) / 2).unsqueeze(-1)
        width = (bbox[:,2]-bbox[:,0]).unsqueeze(-1)
        hight = (bbox[:,3]-bbox[:,1]).unsqueeze(-1)
        #cx = bbox[:,0].unsqueeze(-1)
        #cy = bbox[:,1].unsqueeze(-1)
        #width = bbox[:,2].unsqueeze(-1)
        #hight = bbox[:,3].unsqueeze(-1)
        return torch.stack((cx, cy, width, hight), dim=1), pscore