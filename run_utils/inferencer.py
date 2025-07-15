
import yaml
import os
import logging
import torch
import cv2
import numpy as np
from run_utils.builder import ConfigBuilder
from time import perf_counter
from utils.data_preparation import img_rel_preprocess
from relat_depth_models.load_reldepth_model import load_reldepth_model
import torch.nn.functional as F
from utils.visualization import plot_realat_depth, run_gradcam_on_encoder_img


class Inferencer():
    def __init__(self, args, with_info=False):
        '''
            with_info的作用是什么
        '''
        
        
        cfg_path = args.cfg
        with open(cfg_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

        self.builder = ConfigBuilder(args, **cfg_params)
        self.model_type = self.builder.model_params.get('type', 'tode')
        self.with_info = with_info
        if self.with_info:
            self.logger.info('Building models ...')
        
        # cuda
        self.cuda_id = self.builder.get_inference_cuda_id()
        self.device = torch.device('cuda:{}'.format(self.cuda_id) if torch.cuda.is_available() else 'cpu')
        
        # model
        self.model = self.builder.get_model()
        self.model.to(self.device)
        self.model.eval()
        
        if self.with_info:
            self.logger.info('Checking checkpoints...')
        
        checkpoint_file = self.builder.get_inference_checkpoint_path()
        print(f'loading checkpoint: {checkpoint_file}')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])  #strict=False
            start_epoch = checkpoint['epoch']
            if self.with_info:
                self.logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
        else:
            raise FileNotFoundError('No checkpoint.')
        
        self.image_size = self.builder.get_inference_image_size()
        self.depth_min, self.depth_max = self.builder.get_inference_depth_min_max()
        self.depth_norm = self.builder.get_inference_dpeth_norm()
        
        self.args = args
        
    def inference(self, rgb, depth, rgb_mask=None, target_size=(1280, 720)):
        '''
            Input:
                rgb{}, depth{}, initial RGB and depth image
            
            return:
                the completion depth
        '''
        rgb_rel = rgb.copy()
        
        rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        if rgb_mask is not None:
            rgb_mask = cv2.resize(rgb_mask, self.image_size, interpolation = cv2.INTER_NEAREST)
            # depth = depth * (1-rgb_mask) 
            rgb_mask = torch.FloatTensor(rgb_mask).to(self.device).unsqueeze(0)
            
        depth = np.where(depth < self.depth_min, 0, depth)
        depth = np.where(depth > self.depth_max, 0, depth)
        depth[np.isnan(depth)] = 0
        depth = depth / self.depth_norm
        
        rgb = (rgb / 255.0).transpose(2, 0, 1)
        rgb = torch.FloatTensor(rgb).to(self.device).unsqueeze(0)

        depth = torch.FloatTensor(depth).to(self.device).unsqueeze(0)
        
        if self.args.reldepth_model is not None:
            if self.args.reldepth_model == 'depthanything':
                rgb_relat = img_rel_preprocess(rgb_rel)
            elif self.args.reldepth_model == 'leres':
                rgb_relat = cv2.resize(rgb_rel, (448, 448))
            else:
                raise NameError('no such model')
            rgb_relat = torch.FloatTensor(rgb_relat).to(self.device).unsqueeze(0)
            reldepth_model = load_reldepth_model(self.args.reldepth_model, self.device)
        
        # inference
        with torch.no_grad():
            time_start = perf_counter()
            if self.model_type  == 'yftrans':
                
                if self.args.reldepth_model == 'leres':
                    relative_depth = reldepth_model.inference(rgb_relat.permute(0, 3, 1, 2))
                elif self.args.reldepth_model == 'depthanything':
                    relative_depth = reldepth_model.forward(rgb_relat)[None, :, :, :]
                    # relative_depth.unsqueeze(0)
                # plot_realat_depth(relative_depth)
                # print(relative_depth.shape)
                relative_depth = F.interpolate(relative_depth, size=(480, 640), mode='bilinear', align_corners=False)
                # plot_realat_depth(relative_depth)
                # print(relative_depth.shape)
                depth_res = self.model(rgb, relative_depth, depth, rgb_mask)
                
                # depth_res = self.model(rgb, relative_depth, depth)
                
            elif self.model_type  == 'tode' or 'dfnet':
                depth_res = self.model(rgb, depth)
            else:
                raise NameError('no such mode')
            time_end = perf_counter()
        
        if self.with_info:
            print("Inference finished, time: {:.4f}s.".format(time_end - time_start))
            
        depth_res = depth_res.squeeze(0).squeeze(0).cpu().detach().numpy()
        depth_res = depth_res * self.depth_norm 
        depth_res = cv2.resize(depth_res, target_size, interpolation=cv2.INTER_NEAREST)
        
        
        return depth_res