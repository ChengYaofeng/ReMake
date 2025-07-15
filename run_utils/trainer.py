import torch
import torch.nn.functional as F

#tdcnet
def tdc_trainer(model, data_dict):
    res = model(data_dict['rgb'], data_dict['depth']) #这里预测的结果不一致
    n, h, w = data_dict['depth'].shape
    data_dict['pred'] = res.view(n, h, w)

#dfnet    
def df_trainer(model, data_dict):
    res = model(data_dict['rgb'], data_dict['depth'])
    depth_scale = data_dict['depth_max'] - data_dict['depth_min']
    res = res * depth_scale.reshape(-1, 1, 1) + data_dict['depth_min'].reshape(-1, 1, 1)
    data_dict['pred'] = res

    
#yftrans
def yf_trainer(model, data_dict, relative_depth):
    
    relative_depth = F.interpolate(relative_depth, size=(480, 640), mode='bilinear', align_corners=False)

    # res = model(data_dict['rgb'], relative_depth, data_dict['depth']) #仅仅输入非透明区域的深度
    res = model(data_dict['rgb'], relative_depth, data_dict['depth'], data_dict['depth_gt_mask']) #仅仅输入非透明区域的深度
    
    n, h, w = data_dict['depth'].shape
    data_dict['pred'] = res.view(n, h, w)
    
