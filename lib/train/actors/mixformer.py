from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
available_fonts = [f.name for f in fm.fontManager.ttflist]
print("'serif'" in available_fonts)

class MixFormerActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, objective, loss_weight, settings, run_score_head=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        vis = False
        out_dict, early_out_dict, early_scores = self.forward_pass(data, run_score_head=self.run_score_head)
        vis2d = False
        if vis2d:
            #distill_q_list = out['distill_q_list']
            distill_s_map_list = out_dict['distill_s_map_list']
            distill_feat_list = out_dict['distill_feat_list']

            print(len(distill_s_map_list))
            print(len(distill_feat_list))

            plt.rcParams['font.family'] = 'serif'
#            plt.rcParams['mathtext.fontset'] = 'stix'

#            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_style("whitegrid")
            palette = sns.color_palette("husl", 4)

            
            j = 0
            for i in [1, 2, 3, 4]:
#            for i in [2, 4, 6, 8, 10, 12]:
                fig, axes = plt.subplots(1, 1, figsize=(4, 4))
                print("feat")

                feat_map = distill_feat_list[i - 1][4, 128:128 + 324, 159]
                feat_map_2d = feat_map.reshape(18, 18).cpu().detach().numpy()
                
                vmin = np.percentile(feat_map_2d, 0)
                vmax = np.percentile(feat_map_2d, 100)


#                im1 = axes.imshow(feat_map_2d, cmap='viridis', aspect='auto', vmin=-4.5, vmax=4.5)
                im1 = axes.imshow(feat_map_2d, cmap='viridis', aspect='auto',vmin=vmin, vmax=vmax)
#                axes.set_title(f'L{[2, 4, 6, 8, 12][j]} Feature', fontsize=14, y=-0.13)
                axes.axis('off')
                
                # plt.colorbar(im1, ax=axes[1, j], shrink=0.8)

                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.savefig(f'/data_F/zhouyong/MixFormerV2-main/layer_feature_distilled_{[1, 2, 3, 4][j]}.png', dpi=300, bbox_inches='tight')
                plt.close()
                j += 1
            
            j = 0
            for i in [1, 2, 3, 4]
#            for i in [2, 4, 6, 8, 12]:
                fig2, axes2 = plt.subplots(1, 1, figsize=(4, 4))
                print(distill_s_map_list[i - 1].shape)
                #attn_map = distill_s_map_list[i - 1][4, 10, 168, :-4]
                attn_map = distill_s_map_list[i - 1].mean(dim=1)
                attn_map = attn_map[4, 169, :]
                print(attn_map.shape)
                attn_map_2d = attn_map.reshape(18, 18).cpu().detach().numpy()
                vmin = np.percentile(attn_map_2d, 30)
                vmax = np.percentile(attn_map_2d, 95)

                im0 = axes2.imshow(attn_map_2d, cmap='viridis', aspect='auto',vmin=vmin, vmax=vmax)
                #axes[j].set_title(f'L{[2, 4, 6, 8, 12][j]} Relation', fontsize=14, y=-0.13)
                axes2.axis('off')


                # plt.colorbar(im0, ax=axes[0, j], shrink=0.8)
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                plt.savefig(f'/data_F/zhouyong/MixFormerV2-main/layer_map_distilled_{[1, 2, 3, 4][j]}.png', dpi=300, bbox_inches='tight')
                plt.close()
                j += 1



        
        if vis:
#            distill_q_list = out['distill_q_list']
#            distill_k_list = out['distill_k_list']
#            distill_v_list = out['distill_v_list']
            distill_s_map_list = out_dict['distill_s_map_list']
            distill_feat_list = out_dict['distill_feat_list']
            
            print(len(distill_s_map_list))
            print(len(distill_feat_list))
            
            plt.style.use('seaborn-v0_8-whitegrid')
            #print(plt.style.available)
            sns.set_style("whitegrid")
            palette = sns.color_palette("husl", 4)
            


            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            #fig.suptitle('Distribution of Attention Relations and Features Across Layers', fontsize=16)
            
            j = 0

            for i in [2,4,6,8,12]:
                
                attn_map = distill_s_map_list[i-1][9].cpu().detach().numpy()
                print(attn_map.shape)
                attn_flat = attn_map.flatten()
                x_values = np.arange(len(attn_flat))


                axes[0, j].set_ylim(0, 1)
                
                axes[0, j].plot(x_values, attn_flat, linewidth=1, color=palette[3])
                axes[0, j].set_title(f'Layer {[2,4,6,8,12][j]} Relation Map')
                axes[0, j].set_xlabel('')
                axes[0, j].set_xticks([])
                axes[0, j].set_ylabel('')
                axes[0, j].grid(False)
                j += 1
                #mean_val = np.mean(attn_flat)
                #std_val = np.std(attn_flat)
                #axes[0, i].text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                #   transform=axes[0, i].transAxes, verticalalignment='top',
                #   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            j = 0
            for i in [2,4,6,8,12]:

                feat_map = distill_feat_list[i-1][6].cpu().detach().numpy()
                print(feat_map.shape)

                #feat_mean = np.mean(feat_map[128:128+324], axis=-1)
                feat_mean = feat_map[128:128+324,1]
                
                feat_flat = feat_mean.flatten()
                x_values = np.arange(len(feat_flat))
                
                axes[1, j].set_ylim(-5, 6)

                axes[1, j].bar(x_values, feat_flat, linewidth=1,color=palette[0],edgecolor=palette[0])
                axes[1, j].set_title(f'Layer {[2,4,6,8,12][j]} Feature Value')
                axes[1, j].set_xlabel('')
                axes[1, j].set_xticks([])
                axes[1, j].set_ylabel('')
                axes[1, j].grid(False)
                j += 1
                
                #axes[1, i].text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                #   transform=axes[1, i].transAxes, verticalalignment='top',
                #   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig('/data_F/zhouyong/MixFormerV2-main/layer_distributions.png', dpi=300, bbox_inches='tight')


        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        labels = None
        #if 'pred_scores' in out_dict: #不训练score head
        #    try:
        #        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        #    except:
        #        raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0], labels=labels, early_out_dict=early_out_dict, early_scores=early_scores)
        
        #loss, status = self.compute_losses(out_dict, gt_bboxes[0], labels=labels, early_out_dict=early_out_dict, early_scores=out_dict['pred_scores'])

        return loss, status

    def forward_pass(self, data, run_score_head):
        search_bboxes = box_xywh_to_xyxy(data['search_anno'][0].clone())
        out_dict, early_out_dict, early_scores = self.net(data['template_images'][0], data['template_images'][1], data['search_images'],
                               softmax=True)
#                               run_score_head=run_score_head, gt_bboxes=search_bboxes)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict, early_out_dict, early_scores

    def compute_losses(self, pred_dict, gt_bbox, return_status=True, labels=None, early_out_dict=None, early_scores=None):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute ciou and iou
        try:
            ciou_loss, iou = self.objective['ciou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # weighted sum
        loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss

        early_ious = []
        early_s = []
        if (early_out_dict is not None) and ('pred_scores' not in pred_dict):
            early_total_loss = 0
            for i, early_pred in enumerate(early_out_dict):
                early_pred_boxes = early_pred['pred_boxes']
                if torch.isnan(early_pred_boxes).any():
                    raise ValueError("Network outputs is NAN! Stop Training")
                early_pred_boxes_vec = box_cxcywh_to_xyxy(early_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                try:
                    early_ciou_loss, early_iou = self.objective['ciou'](early_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
                except:
                    early_ciou_loss, early_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                early_ious.append(early_iou)
                # compute l1 loss
                early_l1_loss = self.objective['l1'](early_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
                early_loss = self.loss_weight['ciou'] * early_ciou_loss + self.loss_weight['l1'] * early_l1_loss
                # loss += 1.0*i/(len(early_out_dict)+1) * early_loss
                #loss += 0.2*(i+1)*early_loss
                loss += 1.0 * early_loss
                early_total_loss += 0.8 * early_loss
                
        early_scores = early_scores.squeeze(1).transpose(0, 1)
        #print(early_scores.shape)
        
        # compute cls loss if neccessary
        if ('pred_scores' in pred_dict) and (early_out_dict is not None):
            # score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels) #不训练score head
            mid_loss = 0
            for i, early_pred in enumerate(early_out_dict):
                early_pred_boxes = early_pred['pred_boxes']
                if torch.isnan(early_pred_boxes).any():
                    raise ValueError("Network outputs is NAN! Stop Training")
                early_pred_boxes_vec = box_cxcywh_to_xyxy(early_pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
                try:
                    early_ciou_loss, early_iou = self.objective['ciou'](early_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
                except:
                    early_ciou_loss, early_iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda() #(64,)
                early_ious.append(early_iou)
                early_s.append(early_scores[i])
                # print(early_iou.unsqueeze(-1))
                mid_loss += self.objective['mid_score'](early_scores[i].unsqueeze(-1), early_iou.unsqueeze(-1))
            # loss = score_loss * self.loss_weight['score'] + mid_loss * self.loss_weight['score']
            #score_loss = self.objective['mid_score'](pred_dict['pred_scores'].unsqueeze(-1), iou.unsqueeze(-1))
            score_loss = self.objective['mid_score'](early_scores[3].unsqueeze(-1), iou.unsqueeze(-1))
            #print(pred_dict['pred_scores'])
            #print(iou)
            loss = mid_loss + score_loss
            #loss = mid_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            # mean_score = early_scores[3].mean()
#            if ('pred_scores' not in pred_dict) and (early_out_dict is None):
            early_mean_iou = [early_ious[i].detach().mean() for i in range(len(early_ious))]
            early_mean_s = [early_s[i].detach().mean() for i in range(len(early_s))]
            if ('pred_scores' in pred_dict) and (early_out_dict is not None):
                status = {"Loss/total": loss.item(),
                          #"Loss/scores": score_loss.item(), #不训练score head
                          #"Loss/mid_score": mid_loss.item(),
                          #"IoU(layer3)": early_mean_iou[0].item(),
                          "IoU(layer6)": early_mean_iou[0].item(),
                          #"IoU(layer6)": early_mean_iou[1].item(),
                          "IoU(layer8)": early_mean_iou[1].item(),
                          "IoU(layer10)": early_mean_iou[2].item(),
                          "Score(layer6)": early_mean_s[0].item(),
                          "Score(layer8)": early_mean_s[1].item(),
                          "Score(layer10)": early_mean_s[2].item(),
                          
                          "IoU": mean_iou.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          # "Loss/early_total": early_total_loss.item(),
                          "IoU(layer2)": early_mean_iou[0].item(),
                          "IoU(layer4)": early_mean_iou[1].item(),
                          "IoU(layer6)": early_mean_iou[2].item(),
                          "IoU(layer8)": early_mean_iou[3].item(),
                          "IoU(layer10)": early_mean_iou[4].item(),
                          "IoU(layer12)": mean_iou.item()}
            return loss, status
        else:
            return loss
