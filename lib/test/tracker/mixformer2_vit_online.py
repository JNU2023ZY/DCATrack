from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np
from lib.models.mixformer2_vit import build_mixformer2_vit_online
from lib.models.mixformer2_vit import build_mixformer2_vit_adpt_online
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from PIL import Image
from lib.test.tracker.tracker_utils import vis_attn_maps
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import time
from torch.cuda import amp
from pathlib import Path

def prepare_input(input_size):
    batch_size = 1  # 计算FLOPs通常用batch_size=1
    channels = 3
    height, width = 224, 224  # 根据实际输入尺寸调整
    
    # 生成三个随机图片张量 (模拟实际输入)
    img1 = torch.rand(batch_size, channels, 112, 112).cuda()
    img2 = torch.rand(batch_size, channels, 112, 112).cuda()
    img3 = torch.rand(batch_size, channels, 224, 224).cuda()
    
    # 返回元组：与forward参数顺序一致 (img1, img2, img3)
    return dict(template=img1, online_template=img2, search=img3)


class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        network_fp32 = build_mixformer2_vit_online(params.cfg, train=False)
        network_fp32.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        print(f"Load checkpoint {self.params.checkpoint} successfully!")
        self.cfg = params.cfg
        #self.scaler = amp.GradScaler()

        network_fp32.eval()
        quantize = False
        if quantize:
            self.network = torch.quantization.quantize_dynamic(
                network_fp32, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
        else:
            #self.network = network_fp32.half().cuda()
            #self.network = network_fp32.cpu()
            self.network = network_fp32.cuda()
        #

        self.attn_weights = []
        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "/data_F/zhouyong/MixFormerV2-main/DEBUG_VIS"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
            self.online_size = self.online_sizes[0]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        if hasattr(params, 'online_size'):
            self.online_size = params.online_size
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("Search factor: ", self.params.search_factor)
        print("Update interval is: ", self.update_interval)
        print("Online size is: ", self.online_size)
        print("Max score decay: ", self.max_score_decay)
        print("Model quantized to INT8 with dynamic quantization")

    def initialize(self, image, info: dict):
        # forward the template once
        
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        im = Image.fromarray(z_patch_arr)
        
        #save_path = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/template/template.jpg"
        #im.save(save_path)
        #print(1)      
                           
        if self.params.vis_attn == 1:
            self.z_patch = z_patch_arr
            self.oz_patch = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        
        with torch.no_grad():
            #self.network.set_online(self.template.half(), self.online_template.half())
            self.network.set_online(self.template, self.online_template)
            
#        self.network.forward = self.network.forward_test

#        macs, ps = get_model_complexity_info(self.network, (3, 112, 112), input_constructor=prepare_input,
#                                             as_strings=False, print_per_layer_stat=False, verbose=False)
#        print(f"Model stats: macs: {macs}, and params: {ps}")

        
        self.online_state = info['init_bbox']

        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.max_online_template_arr = z_patch_arr
        self.online_forget_id = 0

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        
        H, W, _ = image.shape
        self.frame_id += 1
        
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        #save_path = os.path.join("/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/search", "%04d.jpg" % self.frame_id)
        #x_patch_arr = cv2.cvtColor(x_patch_arr, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(save_path, x_patch_arr)
                                                                
        search = self.preprocessor.process(x_patch_arr)
        
        
        start_time = time.time()
        with torch.no_grad():
            #out_dict, _ = self.network.forward_test(self.template.half(), self.online_template.half(), search.half(), softmax=True, run_score_head=True)
            out_dict, _ = self.network.forward_test(self.template, self.online_template, search, softmax=True, run_score_head=True)
        
        
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).item()

        
        out_layer = out_dict['out_layer']
        

        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        
        
        update = False
        
        if update:

            self.max_pred_score = self.max_pred_score * self.max_score_decay

            if pred_score > 0.8 and pred_score > self.max_pred_score:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)
                self.online_max_template = self.preprocessor.process(z_patch_arr)
                self.max_pred_score = pred_score
                self.max_online_template_arr = z_patch_arr
            
            if self.frame_id % self.update_interval == 0:
                if self.online_size == 1:
                    self.online_template = self.online_max_template
                    if self.params.vis_attn == 1:
                        self.oz_patch = self.oz_patch_max
                elif self.online_template.shape[0] < self.online_size:
                    self.online_template = torch.cat([self.online_template, self.online_max_template])
                else:
                    self.online_template[self.online_forget_id:self.online_forget_id + 1] = self.online_max_template
                    self.online_forget_id = (self.online_forget_id + 1) % self.online_size

                self.max_pred_score = -1
                self.online_max_template = self.template
            
                #self.network.set_online(self.template, self.online_template)
            
        self.debug = False

        
#        pred_boxes_6 = out_dict['box_6']['pred_boxes'].view(-1, 4)
#        pred_boxes_8 = out_dict['box_8']['pred_boxes'].view(-1, 4)
#        pred_boxes_10 = out_dict['box_10']['pred_boxes'].view(-1, 4)
            
#        pred_box_6 = (pred_boxes_6.mean(dim=0) * self.params.search_size).tolist()
#        pred_box_8 = (pred_boxes_8.mean(dim=0) * self.params.search_size).tolist()
#        pred_box_10 = (pred_boxes_10.mean(dim=0) * self.params.search_size).tolist()
#        pred_box_12 = (pred_boxes.mean(dim=0) * self.params.search_size).tolist()
            
#        pred_box_6_all = (pred_boxes_6.mean(dim=0) * self.params.search_size / resize_factor).tolist()
#        pred_box_8_all = (pred_boxes_8.mean(dim=0) * self.params.search_size / resize_factor).tolist()
#        pred_box_10_all = (pred_boxes_10.mean(dim=0) * self.params.search_size / resize_factor).tolist()

            
#        state_6 = clip_box(self.map_box_back(pred_box_6_all, resize_factor), H, W, margin=10)
#        state_8 = clip_box(self.map_box_back(pred_box_8_all, resize_factor), H, W, margin=10)
#        state_10 = clip_box(self.map_box_back(pred_box_10_all, resize_factor), H, W, margin=10)
        
#        image_mix = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        save_path_mix = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/"
#        save_path = os.path.join(save_path_mix, "%04d.jpg" % self.frame_id)
#        cv2.imwrite(save_path, image_mix)
        
        vis_bbox = False

        if vis_bbox:

            image_mix = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#            image_BGR_6 = x_patch_arr.copy()
#            image_BGR_8 = x_patch_arr.copy()
#            image_BGR_10 = x_patch_arr.copy()
#            image_BGR_12 = x_patch_arr.copy()
            
            image_BGR_6 = image_mix.copy()
            image_BGR_8 = image_mix.copy()
            image_BGR_10 = image_mix.copy()
            image_BGR_12 = image_mix.copy()
            
            
#            image_BGR_6 = cv2.cvtColor(image_BGR_6, cv2.COLOR_BGR2RGB)
#            image_BGR_8 = cv2.cvtColor(image_BGR_8, cv2.COLOR_BGR2RGB)
#            image_BGR_10 = cv2.cvtColor(image_BGR_10, cv2.COLOR_BGR2RGB)
#            image_BGR_12 = cv2.cvtColor(image_BGR_12, cv2.COLOR_BGR2RGB) 

            
            

            save_path_6 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/bbox_6"
            save_path_8 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/bbox_8"
            save_path_10 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/bbox_10"
            save_path_12 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/bbox_12"
            
            save_path_mix = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/bbox_mix"
            
           
            
            x1, y1, w, h = pred_box_6
            x2, y2, w2, h2 = state_6
#            image_BGR_6 = cv2.rectangle(image_BGR_6, (int(x1 - w//2), int(y1 - w //2)), (int(x1 + w //2), int(y1 + h // 2)), color=(0, 0, 255), thickness=2)
            image_BGR_6 = cv2.rectangle(image_BGR_6,(int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)          
            image_mix = cv2.rectangle(image_mix, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(save_path_6, "%04d_6.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR_6)

            x1, y1, w, h = pred_box_8
            x2, y2, w2, h2 = state_8
#            image_BGR_8 = cv2.rectangle(image_BGR_8, (int(x1 - w//2), int(y1 - w //2)), (int(x1 + w //2), int(y1 + h // 2)), color=(0, 0, 255), thickness=2)
            image_BGR_8 = cv2.rectangle(image_BGR_8,(int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)  
            image_mix = cv2.rectangle(image_mix, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(save_path_8, "%04d_8.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR_8)

            x1, y1, w, h = pred_box_10
            x2, y2, w2, h2 = state_10
#            image_BGR_10 = cv2.rectangle(image_BGR_10, (int(x1 - w//2), int(y1 - w //2)), (int(x1 + w //2), int(y1 + h // 2)), color=(0, 0, 255), thickness=2)
            image_BGR_10 = cv2.rectangle(image_BGR_10,(int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)
            image_mix = cv2.rectangle(image_mix, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(save_path_10, "%04d_10.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR_10)
            
            x1, y1, w, h = pred_box_12
            x2, y2, w2, h2 = self.state
#            image_BGR_12 = cv2.rectangle(image_BGR_12, (int(x1 - w//2), int(y1 - w //2)), (int(x1 + w //2), int(y1 + h // 2)), color=(0, 0, 255), thickness=2)
            image_BGR_12 = cv2.rectangle(image_BGR_12,(int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)
            image_mix = cv2.rectangle(image_mix, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), color=(0, 0, 255), thickness=2)
            save_path = os.path.join(save_path_12, "%04d_12.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_BGR_12)
            
            
            save_path = os.path.join(save_path_mix, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image_mix)
            
            
        if self.debug: 
        
            feat_path_6 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/feat_6"
            feat_path_8 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/feat_8"
            feat_path_10 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/feat_10"
            feat_path_12 = "/data_F/zhouyong/MixFormerV2-main/VIS_68910/truck-16/feat_12"
                   
            feat_6 = out_dict['early_feat'][0].squeeze(0).detach().cpu().numpy()[158]
            feat_8 = out_dict['early_feat'][1].squeeze(0).detach().cpu().numpy()[158]
            feat_10 = out_dict['early_feat'][2].squeeze(0).detach().cpu().numpy()[158]
            feat_12 = out_dict['early_feat'][3].squeeze(0).detach().cpu().numpy()[158]
            
            print(feat_6.shape)
            
#            for i in range(768):
#                feat_6i = (feat_6[i] - feat_6[i].min()) / (feat_6[i].max() - feat_6[i].min() + 1e-8) * 255
#                print(feat_6.shape)
#                feat_8i = (feat_8[i] - feat_8[i].min()) / (feat_8[i].max() - feat_8[i].min() + 1e-8) * 255
#                feat_10i = (feat_10[i] - feat_10[i].min()) / (feat_10[i].max() - feat_10[i].min() + 1e-8) * 255
#                feat_12i = (feat_12[i] - feat_12[i].min()) / (feat_12[i].max() - feat_12[i].min() + 1e-8) * 255   
                
#                plt.figure(figsize=(6, 6))
#                plt.imshow(feat_6i, cmap='viridis')
#                plt.axis('off')
#                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#                plt.margins(0, 0)
#                plt.gca().xaxis.set_major_locator(plt.NullLocator())
#                plt.gca().yaxis.set_major_locator(plt.NullLocator())
#                save_path = os.path.join(feat_path_6, "%04d_6.jpg" % i)
#                plt.savefig(save_path, dpi=300, bbox_inches='tight')
#                plt.close()
                             
            
            feat_6 = (feat_6 - feat_6.min()) / (feat_6.max() - feat_6.min() + 1e-8) * 255
            feat_8 = (feat_8 - feat_8.min()) / (feat_8.max() - feat_8.min() + 1e-8) * 255
            feat_10 = (feat_10 - feat_10.min()) / (feat_10.max() - feat_10.min() + 1e-8) * 255
            feat_12 = (feat_12 - feat_6.min()) / (feat_12.max() - feat_12.min() + 1e-8) * 255
            
            
            
            plt.figure(figsize=(6, 6))
            plt.imshow(feat_6, cmap='viridis')
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            save_path = os.path.join(feat_path_6, "%04d_6.jpg" % self.frame_id)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(6, 6))
            plt.imshow(feat_8, cmap='viridis')
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            save_path = os.path.join(feat_path_8, "%04d_8.jpg" % self.frame_id)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(6, 6))
            plt.imshow(feat_10, cmap='viridis')
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            save_path = os.path.join(feat_path_10, "%04d_10.jpg" % self.frame_id)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(6, 6))
            plt.imshow(feat_12, cmap='viridis')
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            save_path = os.path.join(feat_path_12, "%04d_12.jpg" % self.frame_id)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            

        

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state,
                    "bbox_6": self.state,
                    "bbox_8": self.state,
                    "bbox_10": self.state,
                    "all_boxes": all_boxes_save,
                    "inference_time": time.time() - start_time,
                    'out_layer': out_layer,
                    "score": pred_score}
        else:
            return {"target_bbox": self.state,
                    "bbox_6": self.state,
                    "bbox_8": self.state,
                    "bbox_10": self.state,
                    "inference_time": time.time() - start_time,
                    'out_layer': out_layer,
                    "score": pred_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MixFormerOnline