import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import get_surface_normal_from_depth, safe_mean, depth2pc
from utils.visualization import plot_image_process
# from pytorch3d.loss import chamfer_distance
# from extensions.chamfer_distance.chamfer_distance import ChamferDistance
# from time import perf_counter
# from kaolin.metrics.pointcloud import chamfer_distance
# CD = ChamferDistance()
# chamferDist = ChamferDistance()

class Criterion(nn.Module):
    def __init__(self, type, combined_smooth=False, **kwargs):
        super(Criterion, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.type = str.lower(type)
        if 'huber' in self.type:
            self.huber_k = kwargs.get('huber_k', 0.1)
        self.combined_smooth = combined_smooth
        if combined_smooth:
            self.combined_beta = kwargs.get('combined_beta', 0.005)
            self.combined_beta_decay = kwargs.get('combined_beta_decay', 0.1)
            self.combined_beta_decay_milestones = kwargs.get('combined_beta_decay_milestones', [])
            self.cur_epoch = kwargs.get('cur_epoch', 0)
            for milestone in self.combined_beta_decay_milestones:
                if milestone <= self.cur_epoch:
                    self.combined_beta = self.combined_beta * self.combined_beta_decay
        self.l2_loss = self.mse_loss
        self.masked_l2_loss = self.masked_mse_loss
        self.custom_masked_l2_loss = self.custom_masked_mse_loss
        self.main_loss = getattr(self, type)
        self._mse = self._l2
        
    
    def step(self):
        if self.combined_smooth:
            self.cur_epoch += 1
            if self.cur_epoch in self.combined_beta_decay_milestones:
                self.combined_beta = self.combined_beta * self.combined_beta_decay
                
    def _l1(self, pred, gt):
        '''
            l1 loss in pixel-wise representations
        '''
        return torch.abs(pred - gt)
    
    def _l2(self, pred, gt):
        '''
            l2 loss in pixel-wise representations
        '''
        return (pred - gt) ** 2
    
    
    def mse_loss(self, data_dict, *arg, **kwargs):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return safe_mean(self._l2(pred, gt), mask)
    

    def masked_mse_loss(self, data_dict, *args, **kwargs):
        '''
            masked mse
        '''
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return safe_mean(self._l2(pred, gt), mask)

    
    def custom_masked_mse_loss(self, data_dict, *args, **kwargs):
        '''
            custom masked mse loss
        '''
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return safe_mean(self._l2(pred, gt), mask)
    
    
    def l1_loss(self, data_dict, *args, **kwargs):
        '''
            l1 loss
        '''
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return safe_mean(self._l1(pred, gt), mask)
    
    def masked_l1_loss(self, data_dict, *args, **kwargs):
        '''
            masked l1
        '''
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return safe_mean(self._l1(pred, gt), mask)
    
    def custom_masked_l1_loss(self, data_dict, *args, **kwargs):
        '''
            loss mask for l1 loss
        '''
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        loss = safe_mean(self._l1(pred, gt), mask)
        return loss
    
    def smooth_loss(self, data_dict, *args, **kwargs):
        '''
            surface normal loss
        '''
        pred = data_dict['pred']
        fx, fy, cx, cy = data_dict['fx'], data_dict['fy'], data_dict['cx'], data_dict['cy']
        depth_gt_sn = data_dict['depth_gt_sn']
        _, original_h, original_w = data_dict['depth_original'].shape
        mask = data_dict['loss_mask_dilated']
        # calculate smooth loss
        pred_sn = get_surface_normal_from_depth(pred, fx, fy, cx, cy, original_size=(original_w, original_h))
        sn_loss = 1 - F.cosine_similarity(pred_sn, depth_gt_sn, dim=1)
        #masking
        return safe_mean(sn_loss, mask)
    
    # def cd_loss(self, data_dict, use_mask=True, *args, **kwargs):
    #     pred = data_dict['pred']
    #     gt = data_dict['depth_gt']
    #     # start = perf_counter()
    #     fx, fy, cx, cy = data_dict['fx'], data_dict['fy'], data_dict['cx'], data_dict['cy']
    #     if use_mask:
    #         mask = data_dict['depth_gt_mask'] & data_dict['zero_mask']
    #         pred = pred * mask
    #         gt = gt * mask
        
    #     pred_pc = depth2pc(pred, fx, fy, cx, cy)
    #     gt_pc = depth2pc(gt, fx, fy, cx, cy)
    #     # print(f'point cal time:{perf_counter() - start}')
    #     # pred_pc = pred_pc.to(dtype=torch.float32)
    #     # gt_pc = gt_pc.to(dtype=torch.float32)
        
    #     # dist1, dist2 = CD(pred_pc, gt_pc)
    #     # dist1 = torch.sqrt(dist1)
    #     # dist2 = torch.sqrt(dist2)
    #     # chamfer_dist = (torch.mean(dist1) + torch.mean(dist2)) / 2.0
    #     # print(f'all loss time:{perf_counter() - start}')
    #     # print(f'all loss time:{perf_counter() - start}')
    #     cd = chamfer_distance(pred_pc, gt_pc)
    #     return cd.mean()
    
    def forward(self, data_dict):
        loss_dict = {self.type: self.main_loss(data_dict)}
        if self.combined_smooth:
            loss_dict['smooth'] = self.smooth_loss(data_dict)
            # loss_dict['cd'] = self.cd_loss(data_dict)
            # plot_image_process(data_dict['depth'].cpu().detach().numpy(), data_dict['depth_gt'].cpu().detach().numpy(), data_dict['pred'].cpu().detach().numpy(),
            #                    data_dict['rgb'].cpu().detach().numpy(), data_dict['depth_gt_mask'].cpu().detach().numpy(), data_dict['depth_gt_mask_original'].cpu().detach().numpy())
            
            loss_dict['loss'] = loss_dict[self.type] + self.combined_beta * loss_dict['smooth'] #+ loss_dict['cd']
        else:
            loss_dict['loss'] = loss_dict[self.type]
        return loss_dict