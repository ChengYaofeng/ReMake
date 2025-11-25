import torch
import einops
import os
import numpy as np
import torch.nn.functional as F


def gradient(xyz):
    '''
        Get gradient of xyz map
    '''
    left = xyz
    right = F.pad(xyz, [0, 1, 0, 0])[:, :, :, 1:]
    top = xyz
    bottom = F.pad(xyz, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = right - left, bottom - top
    dx[:, :, :, -1] = 0 
    dy[:, :, -1, :] = 0
    
    return dx, dy
    

def get_xyz(depth, fx, fy, cx, cy, original_size=(1280, 720)):
    '''
        get xyz from depth image and camera intrinsics
    '''
    bs, h, w = depth.shape
    indices = np.indices((h, w), dtype=np.float32)
    indices = torch.FloatTensor(np.array([indices] * bs)).to(depth.device)
    x_scale = w / original_size[0]
    y_scale = h / original_size[1]
    fx *= x_scale
    fy *= y_scale
    cx *= x_scale
    cy *= y_scale
    z = depth
    x = (indices[:, 1, :, :] - einops.repeat(cx, 'bs -> bs h w', h = h, w = w)) * z / einops.repeat(fx, 'bs -> bs h w', h = h, w = w)
    y = (indices[:, 0, :, :] - einops.repeat(cy, 'bs -> bs h w', h = h, w = w)) * z / einops.repeat(fy, 'bs -> bs h w', h = h, w = w)
    return torch.stack([x, y, z], axis = 1)


def get_surface_normal_from_xyz(xyz, epsilon=1e-8):
    '''
        get surface normal from xyz map
        
        Input:
            xyz:
        Output:
            surface:
    '''
    dx, dy = gradient(xyz)
    surface_normal = torch.cross(dx, dy, dim=1)
    surface_normal = surface_normal / (torch.norm(surface_normal, dim=1, keepdim=True) + epsilon)
    return surface_normal


def get_surface_normal_from_depth(depth, fx, fy, cx, cy, original_size=(1280, 720), epsilon=1e-8):
    '''
        get surface normal from depth and camera intrinsics
        Input:
            depth:
            fx, fy, cx, cy: tensor
            original_size: tuple
            epsilon: float
        Output:
            surface
    '''
    xyz = get_xyz(depth, fx, fy, cx, cy, original_size)
    return get_surface_normal_from_xyz(xyz, epsilon)


def to_device(data_dict, device):
    """
        to device
    """
    for key in data_dict.keys():
        data_dict[key] = data_dict[key].to(device)
    return data_dict


def display_results(metrics_dict, logger):
    """
    Given a metrics dict, display the results using the logger.

    Parameters
    ----------
        
    metrics_dict: dict, required, the given metrics dict;

    logger: logging.Logger object, the logger.
    """
    try:
        display_list = []
        for key in metrics_dict.keys():
            if key == 'samples':
                num_samples = metrics_dict[key]
            else:
                display_list.append([key, float(metrics_dict[key])])
        logger.info("Metrics on {} samples:".format(num_samples))
        for display_line in display_list:
            metric_name, metric_value = display_line
            logger.info("  {}: {:.6f}".format(metric_name, metric_value))    
    except Exception:
        logger.warning("Unable to display the results, the operation is ignored.")
        pass
    

def safe_mean(data, mask, default_res=0.):
    masked_data = data[mask]
    return torch.tensor(default_res).to(masked_data.device) if masked_data.numel() == 0 else masked_data.mean()


def safe_mean_without_inf(data, default_res = 0.0):
    mask = torch.isfinite(data)
    return safe_mean(data, mask, default_res = default_res)

def depth2pc(depth, fx, fy, cx, cy):
    '''
        Input: 
            depth: [B,H,W]
            intrinsics: np. 
        
    '''
    B, H, W = depth.shape
    device = depth.device
    # print(device)

    # camera intrinsics (fx, fy, cx, cy)
    # cam_intrinsics = np.load('/home/cyf/remake/datasets/transcg/transcg/camera_intrinsics/1-camIntrinsics-D435.npy')
    # fx = cam_intrinsics[0, 0]
    # fy = cam_intrinsics[1, 1]
    # cx = cam_intrinsics[0, 2]
    # cy = cam_intrinsics[1, 2]

    # pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.float32, device=device),
        torch.arange(0, W, dtype=torch.float32, device=device),
        indexing='ij'
    )  # [H, W]

    x = x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    y = y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    d = depth         # [B, H, W]
    
    fx = fx.view(B, 1, 1).expand(B, H, W)
    fy = fy.view(B, 1, 1).expand(B, H, W)
    cx = cx.view(B, 1, 1).expand(B, H, W)
    cy = cy.view(B, 1, 1).expand(B, H, W)
    # projection formula
    X = (x - cx) * d / fx
    Y = (y - cy) * d / fy
    Z = d

    pc = torch.stack((X, Y, Z), dim=1)  # [B, 3, H, W]
    pc = pc.view(B, 3, -1).permute(0, 2, 1)  # [B, N, 3]
    return pc