from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch


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
        out_dict, early_out_dict, early_scores = self.forward_pass(data, run_score_head=self.run_score_head)

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

        return loss, status

    def forward_pass(self, data, run_score_head):
        search_bboxes = box_xywh_to_xyxy(data['search_anno'][0].clone())
        out_dict, early_out_dict, early_scores = self.net(data['template_images'][0], data['template_images'][1], data['search_images'],
                               softmax=True,
                               run_score_head=run_score_head, gt_bboxes=search_bboxes)
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
                #loss += 1.0*i/(len(early_out_dict)+1) * early_loss
                loss += 1.0 * early_loss
                early_total_loss += 1.0 * early_loss

        # compute cls loss if neccessary
        if ('pred_scores' in pred_dict) or (early_out_dict is not None):
            #score_loss = self.objective['score'](pred_dict['pred_scores'].view(-1), labels) #不训练score head
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
                mid_loss += self.objective['mid_score'](early_scores[i].unsqueeze(-1), early_iou.unsqueeze(-1))
            #loss = score_loss * self.loss_weight['score'] + mid_loss * self.loss_weight['score']
            loss = mid_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            if ('pred_scores' not in pred_dict) and (early_out_dict is None):
                early_mean_iou = [early_ious[i].detach().mean() for i in range(len(early_ious))]
            if ('pred_scores' in pred_dict) or (early_out_dict is not None):
                status = {"Loss/total": loss.item(),
                          #"Loss/scores": score_loss.item(), #不训练score head
                          #"Loss/mid_score": mid_loss.item(),
                          "IoU": mean_iou.item()}
            else:
                status = {"Loss/total": loss.item(),
                          "Loss/ciou": ciou_loss.item(),
                          "Loss/l1": l1_loss.item(),
                          "Loss/early_total": early_total_loss.item(),
                          #"IoU(layer2)": early_mean_iou[0].item(),
                          #"IoU(layer4)": early_mean_iou[0].item(),
                          "IoU(layer6)": early_mean_iou[0].item(),
                          "IoU(layer8)": early_mean_iou[1].item(),
                          "IoU(layer10)": early_mean_iou[2].item(),
                          "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
