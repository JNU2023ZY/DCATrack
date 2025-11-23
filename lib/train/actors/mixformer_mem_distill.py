from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixFormerMemDistillActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, objective, loss_weight, settings, net_teacher, run_score_head=False,
                 z_size_teacher=None, x_size_teacher=None, feat_sz=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

        self.net_teacher = net_teacher.eval()
        self.distill_logits_loss = nn.KLDivLoss(reduction="batchmean")
        self.feat_sz = feat_sz
        self.z_size_teacher = z_size_teacher
        self.x_size_teacher = x_size_teacher
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
        out_dict, early_out_dict = self.forward_pass(data, run_score_head=self.run_score_head)
        with torch.no_grad():
            out_dict_teacher = self.forward_pass_teacher(data)
        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        labels = None
        #if 'pred_scores' in out_dict: #不训练score head
        #    try:
        #        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        #    except:
        #        raise Exception("Please setting proper labels for score branch.")

        # compute losses
        loss, status = self.compute_losses(out_dict, out_dict_teacher, gt_bboxes, labels=labels, early_out_dict=early_out_dict)

        return loss, status

    def forward_pass(self, data, run_score_head):
        out_dict, early_out_dict = self.net(data['template_images'][0], data['template_images'][1], data['search_images'],
                               softmax=False,
                               run_score_head=run_score_head,addmem=True)
        return out_dict, early_out_dict

    def forward_pass_teacher(self, data):
        out_dict, _ = self.net_teacher(data['template_images'][0], data['template_images'][1], data['search_images'], softmax=True, addmem=False)
        return out_dict
    def compute_losses(self, pred_dict, pred_dict_teacher, gt_bbox, return_status=True, labels=None, early_out_dict=None, early_scores=None):
        # Get boxes
        # loss = []
        ious = []
        tea_ious = []
        all_early_ious = []
        total_loss = 0
        for i in range(len(pred_dict)):
            pred_boxes = pred_dict[i]['pred_boxes']
            pred_boxes_tea = pred_dict_teacher[i]['pred_boxes']

            if torch.isnan(pred_boxes).any():
                print("pred_boxes")
                print(pred_boxes)
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            pred_boxes_vec_tea = box_cxcywh_to_xyxy(pred_boxes_tea).view(-1, 4)     # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox[i])[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute ciou and iou
            try:
                ciou_loss, iou = self.objective['ciou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                ciou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            try:
                _, iou_teacher = self.objective['ciou'](pred_boxes_vec_tea, gt_boxes_vec)
            except:
                iou_teacher = torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

            distill_loss_logits = self.compute_losses_distill(pred_dict[i], pred_dict_teacher[i])
            # weighted sum
            loss_ = self.loss_weight['ciou'] * ciou_loss + self.loss_weight['l1'] * l1_loss + 5 * distill_loss_logits

            ious.append(iou)
            tea_ious.append(iou_teacher)

            early_ious = []
            if (early_out_dict[i] is not None) and ('pred_scores' not in pred_dict[i]):
                for k, early_pred in enumerate(early_out_dict[i]):
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
                    early_logits_loss = self.compute_losses_distill(early_pred, pred_dict_teacher[i])
                    early_loss = self.loss_weight['ciou'] * early_ciou_loss + self.loss_weight['l1'] * early_l1_loss + 5 * early_logits_loss
                    loss_ += (0.2 + 0.2*(k+1))*early_loss

            total_loss += loss_
            all_early_ious.append(early_ious)



        if return_status:
            # status for log
            mean_iou1 = ious[0].detach().mean()
            mean_iou2 = ious[1].detach().mean()
            mean_iou3 = ious[2].detach().mean()
            mean_tea_iou1 = tea_ious[0].detach().mean()
#            if ('pred_scores' not in pred_dict) and (early_out_dict is None):
            early_mean_iou1 = [all_early_ious[0][i].detach().mean() for i in range(len(all_early_ious[0]))]
            #early_mean_iou2 = [all_early_ious[1][i].detach().mean() for i in range(len(all_early_ious[1]))]

            status = {"Totalloss": total_loss.item(),
                      # "Loss/S1": loss[0].item(),
                      # "Loss/S2": loss[1].item(),
                      # "Loss/ciou": ciou_loss.item(),
                      # "Loss/l1": l1_loss.item(),
                      # "Loss/early_total": early_total_loss.item(),
                      "IoU_1(layer6)": early_mean_iou1[0].item(),
                      "IoU_1(layer8)": early_mean_iou1[1].item(),
                      "IoU_1(layer10)": early_mean_iou1[2].item(),
                      "IoU_1(layer12)": mean_iou1.item(),
                      # "IoU_2(layer6)": early_mean_iou2[0].item(),
                      # "IoU_2(layer8)": early_mean_iou2[1].item(),
                      # "IoU_2(layer10)": early_mean_iou2[2].item(),
                      "IoU_2(layer12)": mean_iou2.item(),
                      "IoU_3(layer12)": mean_iou3.item(),
                      "Tea_IoU_1(layer12)": mean_tea_iou1.item(),


            }
            # mem_stats = torch.cuda.memory_stats()
            return total_loss, status
        else:
            return total_loss


    def compute_losses_distill(self, pred_dict, pred_dict_teacher):
        """
        prob_tl/br: corner logits before softmax for student, prob after softmax for teacher, shape (b, hw)
        distill_feat_list: features, shape (b, hw, c)
        """
        prob_l = pred_dict['prob_l']
        prob_r = pred_dict['prob_r']
        prob_t = pred_dict['prob_t']
        prob_b = pred_dict['prob_b']    # (b, feat_sz)

        feat_sz = self.feat_sz
        # assert feat_sz ** 2 == pred_dict_teacher['prob_tl'].size(1)
        # ptl_tea, pbr_tea = pred_dict_teacher['prob_tl'].detach(), pred_dict_teacher['prob_br'].detach()
        # ptl_tea = ptl_tea.view((-1, feat_sz, feat_sz))
        # pbr_tea = pbr_tea.view((-1, feat_sz, feat_sz))
        # prob_t_tea = ptl_tea.sum(dim=2)
        # prob_l_tea = ptl_tea.sum(dim=1)
        # prob_b_tea = pbr_tea.sum(dim=2)
        # prob_r_tea = pbr_tea.sum(dim=1) # (b, feat_sz)
        prob_t_tea = pred_dict_teacher['prob_t'].detach()
        prob_l_tea = pred_dict_teacher['prob_l'].detach()
        prob_b_tea = pred_dict_teacher['prob_b'].detach()
        prob_r_tea = pred_dict_teacher['prob_r'].detach()  # (b, feat_sz)

        dis_loss_logits = (self.distill_logits_loss(F.log_softmax(prob_t, dim=1), prob_t_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_l, dim=1), prob_l_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_b, dim=1), prob_b_tea) +
                           self.distill_logits_loss(F.log_softmax(prob_r, dim=1), prob_r_tea)) / 4

        return dis_loss_logits