import torch
import numpy as np

def evaluate_transparent_region_ratio(data_dict, epsilon=2e-3):
    """
    calculate different region ratios in the transparent area.

    parameters:
    - data_dict: data in batch (have to contain: depth, depth_gt, depth_mask)
    - epsilon: 判断正常区域的误差容忍(默认 0.0001)

    return:
    - ratio(refract_ratio, reflect_ratio, normal_ratio)
    - pixel_num(refract_count, reflect_count, normal_count)
    """
    depth = data_dict["depth"]        # [B,1,H,W] or [B,H,W]
    depth_gt = data_dict["depth_gt"]
    mask = data_dict["depth_gt_mask"]

    if depth.dim() == 4:
        depth = depth[:, 0]
        depth_gt = depth_gt[:, 0]
        mask = mask[:, 0]

    B = depth.shape[0]
    total_refract = total_reflect = total_normal = 0

    for i in range(B):
        d = depth[i].cpu().numpy()
        d_gt = depth_gt[i].cpu().numpy()
        d_mask = mask[i].cpu().numpy()
        d_err = np.abs(d - d_gt)

        trans_mask = (d_mask == 1)
        reflect_mask = (d == 0) & trans_mask
        normal_mask = (d != 0) & (d_err < epsilon) & trans_mask
        refract_mask = ~(reflect_mask | normal_mask) & trans_mask

        total_reflect += np.count_nonzero(reflect_mask)
        total_normal += np.count_nonzero(normal_mask)
        total_refract += np.count_nonzero(refract_mask)

    total_pixels = total_refract + total_reflect + total_normal
    if total_pixels == 0:
        return (0, 0, 0), (0, 0, 0)

    refract_ratio = total_refract / total_pixels
    reflect_ratio = total_reflect / total_pixels
    normal_ratio = total_normal / total_pixels

    return (refract_ratio, reflect_ratio, normal_ratio), (total_refract, total_reflect, total_normal)


# def evaluate_model_statistically(data_dict, error_thresh=0.05):
#     pred_depth = data_dict['pred']
#     rgb = data_dict['rgb']
#     depth = data_dict['depth']
#     depth_gt = data_dict['depth_gt']
#     mask = data_dict['depth_gt_mask']

#     abs_error = torch.abs(pred_depth - depth_gt)

#     batch_refract_errors = []
#     batch_reflect_errors = []

#     for i in range(rgb.shape[0]):
#         d_in = depth[i, 0]
#         # d_gt = depth_gt[i, 0]
#         d_mask = mask[i, 0]
#         d_err = abs_error[i, 0]

#         trans_mask = d_mask == 1
#         reflect_mask = (d_in == 0) & trans_mask
#         refract_mask = (d_in != 0) & trans_mask & (d_err < error_thresh)

#         refract_error = d_err[refract_mask].cpu().numpy()
#         reflect_error = d_err[reflect_mask].cpu().numpy()

#         if len(refract_error) > 0:
#             batch_refract_errors.append(refract_error)
#         if len(reflect_error) > 0:
#             batch_reflect_errors.append(reflect_error)

#     return batch_refract_errors, batch_reflect_errors

def evaluate_model_statistically(data_dict, error_thresh=0.05, epsilon=1e-3):
    """
    error statistics for different transparent regions: refraction, reflection, normal.
    
    - error_thresh: maximum error to filter outliers
    - epsilon: threshold to determine normal region
    """
    pred_depth = data_dict['pred']
    rgb = data_dict['rgb']
    depth = data_dict['depth']
    depth_gt = data_dict['depth_gt']
    mask = data_dict['depth_gt_mask']

    abs_error = torch.abs(pred_depth - depth_gt)

    batch_refract_errors = []
    batch_reflect_errors = []
    batch_normal_errors = []

    for i in range(rgb.shape[0]):
        d_in = depth[i, 0]
        d_gt = depth_gt[i, 0]
        d_mask = mask[i, 0]
        d_err = abs_error[i, 0]

        trans_mask = d_mask == 1
        reflect_mask = (d_in == 0) & trans_mask
        normal_mask = (d_in != 0) & trans_mask & (torch.abs(d_in - d_gt) < epsilon)
        refract_mask = trans_mask & ~(reflect_mask | normal_mask)

        # filter outliers
        refract_error = d_err[refract_mask & (d_err < error_thresh)].cpu().numpy()
        reflect_error = d_err[reflect_mask & (d_err < error_thresh)].cpu().numpy()
        normal_error = d_err[normal_mask & (d_err < error_thresh)].cpu().numpy()

        if len(refract_error) > 0:
            batch_refract_errors.append(refract_error)
        if len(reflect_error) > 0:
            batch_reflect_errors.append(reflect_error)
        if len(normal_error) > 0:
            batch_normal_errors.append(normal_error)

    return batch_refract_errors, batch_reflect_errors, batch_normal_errors
