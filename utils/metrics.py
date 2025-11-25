import logging
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from monai.metrics.regression import SSIMMetric
from monai.losses.ssim_loss import SSIMLoss
from utils.tools import display_results
from run_utils.logger import get_logger
# from pytorch3d.loss import chamfer_distance
import numpy as np
from utils.tools import depth2pc

class Metrics():
    def __init__(self, epsilon=1e-8, depth_scale=1.0, **kwargs):
        super(Metrics, self).__init__()
        self.epsilon = epsilon
        self.depth_scale = depth_scale
        
    def MSE(self, pred, gt, zero_mask, *args, **kwargs):
        '''
            
        '''
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim=[1, 2]) / (
            torch.sum(zero_mask.float(), dim=[1, 2]) + self.epsilon
        ) * self.depth_scale * self.depth_scale
        return torch.mean(sample_mse).item()
    
    def RMSE(self, pred, gt, zero_mask, *args, **kwargs):
        '''
        Root Mean Square Error
        '''
        sample_mse = torch.sum(((pred - gt) ** 2) * zero_mask.float(), dim=[1, 2]) / (
            torch.sum(zero_mask.float(), dim=[1, 2]) + self.epsilon) * self.depth_scale * self.depth_scale
        return torch.mean(torch.sqrt(sample_mse)).item()
    
    
    def MaskedMSE(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        '''
        
        '''
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim= [1, 2]) / (
            torch.sum(mask.float(), dim=[1, 2]) + self.epsilon) * self.depth_scale * self.depth_scale
        return torch.mean(sample_masked_mse).item()
    
    
    def MaskedRMSE(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        '''
        Masked Root Mean Square Error
        '''
        mask = gt_mask & zero_mask
        sample_masked_mse = torch.sum(((pred - gt) ** 2) * mask.float(), dim= [1, 2]) / (
            torch.sum(mask.float(), dim=[1, 2]) + self.epsilon) * self.depth_scale * self.depth_scale
        return torch.mean(torch.sqrt(sample_masked_mse)).item()
    
    
    def REL(self, pred, gt, zero_mask, *args, **kwargs):
        '''
        relative error
        '''
        sample_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * zero_mask.float(), dim=[1,2]) / (
            torch.sum(zero_mask.float(), dim=[1, 2]) + self.epsilon)
        return torch.mean(sample_rel).item()
    
    
    def MaskedREL(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        '''
        Masked relative error
        '''
        mask = gt_mask & zero_mask
        sample_masked_rel = torch.sum((torch.abs(pred - gt) / (gt + self.epsilon)) * mask.float(), dim = [1, 2]) / (
            torch.sum(mask.float(), dim = [1, 2]) + self.epsilon)
        return torch.mean(sample_masked_rel).item()
    
    
    def MAE(self, pred, gt, zero_mask, *args, **kwargs):
        '''
        Mean Absolute Error
        '''
        sample_mae = torch.sum(torch.abs(pred - gt) * zero_mask.float(), dim = [1, 2]) / (
            torch.sum(zero_mask.float(), dim = [1, 2]) + self.epsilon) * self.depth_scale
        return torch.mean(sample_mae).item()
    
    
    def MaskedMAE(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        '''
        masked mae metric
        '''
        mask = gt_mask & zero_mask
        sample_masked_mae = torch.sum(torch.abs(pred - gt) * mask.float(), dim = [1, 2]) / (
            torch.sum(mask.float(), dim = [1, 2]) + self.epsilon) * self.depth_scale
        return torch.mean(sample_masked_mae).item()
    
    
    def Threshold(self, pred, gt, zero_mask, *args, **kwargs):
        delta = kwargs.get('delta', 1.25)
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & zero_mask).float().sum(dim = [1, 2]) / (torch.sum(zero_mask.float(), dim = [1, 2]) + self.epsilon)
        return torch.mean(res).item() * 100
    
    def MaskedThreshold(self, pred, gt, zero_mask, gt_mask, *args, **kwargs):
        delta = kwargs.get('delta', 1.25)
        mask = gt_mask & zero_mask
        thres = torch.maximum(pred / (gt + self.epsilon), gt / pred)
        res = ((thres < delta) & mask).float().sum(dim = [1, 2]) / (torch.sum(mask.float(), dim = [1, 2]) + self.epsilon)
        return torch.mean(res).item() * 100
    
    # def ChamferDist(selg, pred, gt, zero_mask, gt_mask, *args, **kwargs):
    #     mask = gt_mask & zero_mask
    #     mask_pred = depth2pc(pred*mask)
    #     mask_gt = depth2pc(gt*mask)
    #     chamfer_dist, _ = chamfer_distance(mask_pred, mask_gt)
    #     return chamfer_dist

    def ssim(self, pred, gt, zero_mask, gt_mask, **metric_kwargs):

        self.ssim = SSIMMetric(spatial_dims=2,win_size=5,data_range=1.0,reduction='sum')(pred.unsqueeze(1), gt.unsqueeze(1))
        meanssim = torch.mean(self.ssim).item()
        # n,_,_,_ = pred.shape
        sumssim = torch.sum(self.ssim)
        return meanssim


class MetricsRecorder():
    def __init__(self, logger_name, metrics_list, epsilon=1e-8, depth_scale=10.0, **kwargs):
        '''
        Input:
            metrics_list: str list of metrics
        '''
        super(MetricsRecorder, self).__init__()
        self.logger = get_logger(logger_name)
        self.epsilon = epsilon
        self.depth_scale = depth_scale
        self.metrics = Metrics(epsilon=epsilon, depth_scale=depth_scale)
        self.metrics_list = []
        for metric in metrics_list:
            try:
                if "Threshold@" in metric:
                    split_list = metric.split('@')
                    if len(split_list) != 2:
                        raise ArithmeticError('Invalid Metric')
                    delta = float(split_list[1])
                    metric_func = getattr(self.metrics, split_list[0])
                    self.metrics_list.append([metric, metric_func, {'delta': delta}])
                else:
                    # Other metrics.
                    metric_func = getattr(self.metrics, metric)
                    self.metrics_list.append([metric, metric_func, {}])

            except Exception:
                self.logger.warning()
                pass
        self._clear_recorder_dict()


    def clear(self):
        '''
            clear record dict
        '''
        self._clear_recorder_dict()
        
    def _clear_recorder_dict(self):
        '''
            clear metric recorder
        '''
        self.metrics_recorder_dict = {}
        for metric_line in self.metrics_list:
            metric_name, _, _ = metric_line
            self.metrics_recorder_dict[metric_name] = 0
        self.metrics_recorder_dict['samples'] = 0
        
        
    def _update_recorder_dict(self, metrics_dict):
        '''
            update metric dict with a batch of samples
        '''
        for metric_line in self.metrics_list:
            metric_name, _, _ = metric_line
            self.metrics_recorder_dict[metric_name] += metrics_dict[metric_name] * metrics_dict['samples']
        self.metrics_recorder_dict['samples'] += metrics_dict['samples']
        
        
    def evaluate_batch(self, data_dict, record=True, original=False, *args, **kwargs):
        '''
            evaluate a batch of samples
        '''
        resize1 = transforms.Resize([144, 256], interpolation=InterpolationMode.NEAREST)
        resize2 = transforms.Resize([144, 256], interpolation=InterpolationMode.NEAREST)
        pred = resize1(data_dict['pred'])
        if not original:
            gt = resize1(data_dict['depth_gt'])
            zero_mask = resize2(data_dict['zero_mask'])
            gt_zero_mask = gt > 0
            zero_mask = torch.logical_and(zero_mask, gt_zero_mask)
            gt_mask = resize2(data_dict['depth_gt_mask'])
            gt_mask = torch.logical_and(gt_mask, gt_zero_mask)
        else:
            gt = resize1(data_dict['depth_gt_original'])
            gt_mask = resize1(data_dict['depth_gt_mask_original'])
            zero_mask = resize2(data_dict['zero_mask_original'])
        num_samples = gt.shape[0]
        metrics_dict = {'samples': num_samples}
        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            metrics_dict[metric_name] = metric_func(pred, gt, zero_mask, gt_mask, **metric_kwargs)
        if record:
            self._update_recorder_dict(metrics_dict)
        return metrics_dict

    
    def get_results(self):
        '''
            get the final results of metrics
        '''
        final_metrics_dict = self.metrics_recorder_dict.copy()
        for metric_line in self.metrics_list:
            metrics_name, _, _ = metric_line
            final_metrics_dict[metrics_name] /= final_metrics_dict['samples']
        return final_metrics_dict
    
    def display_recorder_results(self):
        '''
            display the metrics recorder dict
        '''
        display_results(self.get_results(), self.logger)