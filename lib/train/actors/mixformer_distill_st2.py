from . import BaseActor
from lib.utils.misc import is_main_process
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixFormerDistillStage2Actor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, objective, loss_weight, settings, net_teacher, run_score_head=False,
                 distill_layers_student=[], distill_layers_teacher=[]):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

        # distill related
        self.net_teacher = net_teacher.eval()
        self.distill_logits_loss = nn.KLDivLoss(reduction="batchmean")
        self.distill_attn_loss = nn.CrossEntropyLoss()
        self.distill_layers_student = distill_layers_student
        self.distill_layers_teacher = distill_layers_teacher
        if is_main_process():
            print(f"Supervise student's {self.distill_layers_student}-th layers with teacher's {self.distill_layers_teacher}-th layers")

    def __call__(self, data, remove_rate_cur_epoch=1):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward student
        out_dict, early_out_dict, early_scores = self.forward_pass(data, remove_rate_cur_epoch)  # forward teacher
        with torch.no_grad():
            out_dict_teacher, early_out_dict_tea = self.forward_pass_teacher(data)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        labels = None
        if 'pred_scores' in out_dict:
            try:
                labels = data['label'].view(-1)  # (batch, ) 0 or 1
            except:
                raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, out_dict_teacher, gt_bboxes[0], labels=labels, early_out_dict=early_out_dict, early_out_dict_tea=early_out_dict_tea)

        return loss, status

    def forward_pass(self, data, remove_rate_cur_epoch=1.0):
        out_dict, early_out_dict, _ = self.net(data['template_images'][0], data['template_images'][1], data['search_images'],
                            softmax=False, remove_rate=remove_rate_cur_epoch)
        return out_dict, early_out_dict, None

    def forward_pass_teacher(self, data):
        out_dict, early_out_dict, _ = self.net_teacher(data['template_images'][0], data['template_images'][1], data['search_images'],
                                    softmax=True, remove_rate=1.0)
        return out_dict, early_out_dict

    def compute_losses(self, pred_dict, pred_dict_teacher, gt_bbox, return_status=True, labels=None, early_out_dict=None, early_out_dict_tea=None):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes_teacher = pred_dict_teacher['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_teacher = box_cxcywh_to_xyxy(pred_boxes_teacher).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)

        # compute ciou and iou
        try:
            ciou_loss, iou = self.objective['ciou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        try:
            _, iou_teacher = self.objective['ciou'](pred_boxes_vec_teacher, gt_boxes_vec)
        except:
            iou_teacher = torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute distillation loss
        distill_loss_logits, distill_loss_feat, distill_loss_mt_map, distill_loss_s_map, dis_loss_q, dis_loss_k, dis_loss_v = self.compute_losses_distill(pred_dict, pred_dict_teacher)

       # weighted sum
        loss = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss + \
               self.loss_weight['corner'] * distill_loss_logits + \
               self.loss_weight['feat'] * distill_loss_feat + (0.00 * distill_loss_mt_map + 0.00 * distill_loss_s_map + 0.005 * dis_loss_v + 0.005 * dis_loss_q + 0.005 * dis_loss_k) * 0.5
        early_ious = []
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
                early_distill_loss_logits = self.compute_losses_logits_distill(early_pred, pred_dict_teacher)
                # early_distill_loss_logits = self.compute_losses_logits_distill(early_pred, early_out_dict_tea[i+1] if i<2 else pred_dict_teacher)
                # print(i*(self.distill_layers_teacher[i]-5)//2)
                #early_distill_loss_logits = self.compute_losses_logits_distill(early_pred, early_out_dict_tea[(self.distill_layers_teacher[i]-5)//2] if i < 2 else pred_dict_teacher)
                self.distill_layers_teacher
                early_ious.append(early_iou)
                # compute l1 loss
                early_l1_loss = self.objective['l1'](early_pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
                early_loss = self.loss_weight['ciou'] * early_ciou_loss + self.loss_weight['l1'] * early_l1_loss + self.loss_weight['corner'] * early_distill_loss_logits
                
                # loss += 1.0*i/(len(early_out_dict)+1) * early_loss
                loss += early_loss
                # loss += 1.0 * early_loss
                early_total_loss += early_loss

        # compute cls loss if neccessary
        if 'pred_scores' in pred_dict:
            score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels)
            loss = score_loss * self.loss_weight['score']

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            early_mean_iou = [early_ious[i].detach().mean() for i in range(len(early_ious))]
            mean_iou_teacher = iou_teacher.detach().mean()
            if 'pred_scores' in pred_dict:
                status = {"Loss/total": loss.item(),
                          "Loss/scores": score_loss.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "IoU(layer6)": early_mean_iou[0].item(),
                          "IoU(layer8)": early_mean_iou[1].item(),
                          "IoU(layer10)": early_mean_iou[2].item(),
                          "Loss/distill_corner_kl": distill_loss_logits.item(),
                          "Loss/distill_feat": distill_loss_feat.item(),
                          #"Loss/distill_mt": distill_loss_mt_map.item(),
                          #"Loss/distill_s": distill_loss_s_map.item(),
                          "Loss/distill_Q": dis_loss_q.item(),
                          "Loss/distill_K": dis_loss_k.item(),
                          "Loss/distill_V": dis_loss_v.item(),
                          "IoU": mean_iou.item(),
                          "IoU_teacher": mean_iou_teacher.item()}
            return loss, status
        else:
            return loss

    def compute_losses_distill(self, pred_dict, pred_dict_teacher):
        """
        prob_l/r/t/b: shape (b, h), before softmax for student, after softmax for teacher
        distill_feat_list: features, shape (b, hw, c)
        """
        prob_l = pred_dict['prob_l']
        prob_r = pred_dict['prob_r']
        prob_t = pred_dict['prob_t']
        prob_b = pred_dict['prob_b']    # (b, feat_sz)

        prob_l_tea = pred_dict_teacher['prob_l'].detach()
        prob_r_tea = pred_dict_teacher['prob_r'].detach()
        prob_t_tea = pred_dict_teacher['prob_t'].detach()
        prob_b_tea = pred_dict_teacher['prob_b'].detach()    # (b, feat_sz)
        dis_loss_logits = 0

        assert prob_l.shape == prob_l_tea.shape
        dis_loss_logits = (self.distill_logits_loss(F.log_softmax(prob_t, dim=1), prob_t_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_l, dim=1), prob_l_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_b, dim=1), prob_b_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_r, dim=1), prob_r_tea))  / 4


        # assert len(pred_dict['distill_feat_list']) == 8 and len(pred_dict_teacher['distill_feat_list']) == 12
        index_s = self.distill_layers_student
        index_t = self.distill_layers_teacher
        dist_feat_stu = torch.stack([pred_dict['distill_feat_list'][i] for i in index_s], dim=0)
        dist_feat_tea = torch.stack([pred_dict_teacher['distill_feat_list'][i].detach() for i in index_t], dim=0)
        
        dist_mt_map_stu = torch.stack([pred_dict['distill_mt_map_list'][i] for i in index_s], dim=0)
        dist_mt_map_tea = torch.stack([pred_dict_teacher['distill_mt_map_list'][i].detach() for i in index_t], dim=0)

        dist_s_map_stu = torch.stack([pred_dict['distill_s_map_list'][i] for i in index_s], dim=0)
        dist_s_map_tea = torch.stack([pred_dict_teacher['distill_s_map_list'][i].detach() for i in index_t], dim=0)
        
        dist_q_stu = torch.stack([pred_dict['distill_q_list'][i] for i in index_s], dim=0)
        dist_q_tea = torch.stack([pred_dict_teacher['distill_q_list'][i].detach() for i in index_t], dim=0)
        
        dist_k_stu = torch.stack([pred_dict['distill_k_list'][i] for i in index_s], dim=0)
        dist_k_tea = torch.stack([pred_dict_teacher['distill_k_list'][i].detach() for i in index_t], dim=0)
        
        dist_v_stu = torch.stack([pred_dict['distill_v_list'][i] for i in index_s], dim=0)
        dist_v_tea = torch.stack([pred_dict_teacher['distill_v_list'][i].detach() for i in index_t], dim=0)
        
        dis_loss_mt_map = 0.0
        dis_loss_s_map = 0.0
        dis_loss_q = 0.0
        dis_loss_k = 0.0
        dis_loss_v = 0.0
        dis_loss_feat = 0.0
        for i in range(len(dist_mt_map_stu)):
            #for stu, tea in zip(dist_mt_map_stu[i], dist_mt_map_tea[i]):
            #    dis_loss_mt_map += self.distill_logits_loss(F.log_softmax(stu, dim=-1),
            #                                           tea.softmax(dim=-1)) / 12
                

            #for stu, tea in zip(dist_s_map_stu[i], dist_s_map_tea[i]):
            #    dis_loss_s_map += self.distill_logits_loss(F.log_softmax(stu, dim=-1),
            #                                           tea.softmax(dim=-1)) / 12
            
            
            for stu, tea in zip(dist_q_stu[i], dist_q_tea[i]):
                stu_qr = torch.matmul(stu, stu.transpose(-2, -1)) / torch.sqrt(torch.tensor(64, dtype=torch.float32))
                tea_qr = torch.matmul(tea, tea.transpose(-2, -1)) / torch.sqrt(torch.tensor(64, dtype=torch.float32))
                dis_loss_q += self.distill_logits_loss(F.log_softmax(stu_qr, dim=-1),
                                                       tea_qr.softmax(dim=-1)) / 12
                                                       
            for stu, tea in zip(dist_k_stu[i], dist_k_tea[i]):
                stu_kr = torch.matmul(stu, stu.transpose(-2, -1)) / torch.sqrt(torch.tensor(64, dtype=torch.float32))
                tea_kr = torch.matmul(tea, tea.transpose(-2, -1)) / torch.sqrt(torch.tensor(64, dtype=torch.float32))
                dis_loss_k += self.distill_logits_loss(F.log_softmax(stu_kr, dim=-1),
                                                       tea_kr.softmax(dim=-1)) / 12
                                                                                                      
            for stu, tea in zip(dist_v_stu[i], dist_v_tea[i]):
                stu_vr = torch.matmul(stu, stu.transpose(-2, -1)) / torch.sqrt(torch.tensor(64, dtype=torch.float32))
                tea_vr = torch.matmul(tea, tea.transpose(-2, -1)) / torch.sqrt(torch.tensor(64, dtype=torch.float32))
                dis_loss_v += self.distill_logits_loss(F.log_softmax(stu_vr, dim=-1),
                                                       tea_vr.softmax(dim=-1)) / 12
            
            #dis_loss_feat += F.mse_loss(dist_feat_stu[i], dist_feat_tea[i])
        dis_loss_feat += F.mse_loss(dist_feat_stu[3], dist_feat_tea[3])
                                                                                                                                                                    
        # dis_loss_mt_map += self.distill_attn_loss(dist_mt_map_tea.softmax(dim=-1), dist_mt_map_stu.softmax(dim=-1))
        # dis_loss_s_map += self.distill_attn_loss(dist_s_map_tea.softmax(dim=-1), dist_s_map_stu.softmax(dim=-1))
        # for stu, tea in zip(dist_mt_map_stu.softmax(dim=-1), dist_mt_map_tea.softmax(dim=-1)):
        #     dis_loss_mt_map += self.distill_attn_loss(stu, tea) / 12
        #
        # for stu, tea in zip(dist_s_map_stu, dist_s_map_tea):
        #     dis_loss_s_map += self.distill_attn_loss(stu.softmax(dim=-1), tea.softmax(dim=-1)) / 12

        # dis_loss_feat = F.mse_loss(dist_feat_stu[3], dist_feat_tea[3])

#        dis_loss_mt_map = F.mse_loss(dist_mt_map_stu, dist_mt_map_tea)
#        dis_loss_s_map = F.mse_loss(dist_s_map_stu, dist_s_map_tea)
        #dis_loss_v = 0.0

        return dis_loss_logits, dis_loss_feat, dis_loss_mt_map, dis_loss_s_map, dis_loss_q, dis_loss_k, dis_loss_v
        
    def compute_losses_logits_distill(self, pred_dict, pred_dict_teacher):
        """
        prob_l/r/t/b: shape (b, h), before softmax for student, after softmax for teacher
        distill_feat_list: features, shape (b, hw, c)
        """
        prob_l = pred_dict['prob_l']
        prob_r = pred_dict['prob_r']
        prob_t = pred_dict['prob_t'] 
        prob_b = pred_dict['prob_b']    # (b, feat_sz)

        prob_l_tea = pred_dict_teacher['prob_l'].detach()
        prob_r_tea = pred_dict_teacher['prob_r'].detach()
        prob_t_tea = pred_dict_teacher['prob_t'].detach()
        prob_b_tea = pred_dict_teacher['prob_b'].detach()    # (b, feat_sz)
        
        dis_loss_logits = 0

        assert prob_l.shape == prob_l_tea.shape
        dis_loss_logits = (self.distill_logits_loss(F.log_softmax(prob_t, dim=1), prob_t_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_l, dim=1), prob_l_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_b, dim=1), prob_b_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_r, dim=1), prob_r_tea))  / 4


        return dis_loss_logits
