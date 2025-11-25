import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2
import matplotlib
import torch
from matplotlib import colors
import pandas as pd

# from .data_preparation import handle_depth
def light_plot_image_process(depth, depth_gt, rgb, rgb_mask):
    """
    Visualize RGB image, original depth image, ground truth depth image, and RGB mask.
    The input image size can vary, and will be resized to 320x240 for display.
    """
    # Copy original data
    rgb_vis = cv2.resize(rgb.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)
    depth_vis = cv2.resize(depth.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)
    depth_gt_vis = cv2.resize(depth_gt.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)
    mask_vis = cv2.resize(rgb_mask.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)

    # Process invalid values
    depth_gt_vis[np.isnan(depth_gt_vis)] = 0.0
    mask_vis = np.where(depth_gt_vis < 1e-7, 1, 0).astype(np.uint8)  # Transparent area is 1

    # Remove unreasonable depth (e.g. > 5m)
    depth_vis[depth_gt_vis > 5] = 0
    depth_gt_vis[depth_gt_vis > 5] = 0

    # Clip and normalize to [0, 255] for display
    depth_vis = np.clip(depth_vis, 0, 1) * 255
    depth_gt_vis = np.clip(depth_gt_vis, 0, 1) * 255

    depth_vis = depth_vis.astype(np.uint8)
    depth_gt_vis = depth_gt_vis.astype(np.uint8)

    # Custom mask display: 1 = red, 0 = black
    mask_cmap = colors.ListedColormap(['black', 'red'])
    mask_bounds = [0, 0.5, 1]
    mask_norm = colors.BoundaryNorm(mask_bounds, mask_cmap.N)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].imshow(rgb_vis[..., ::-1])  # BGR to RGB
    axs[0, 0].set_title("RGB")
    axs[0, 0].axis("off")

    im1 = axs[0, 1].imshow(depth_vis, cmap='jet')
    axs[0, 1].set_title("Original Depth")
    axs[0, 1].axis("off")
    fig.colorbar(im1, ax=axs[0, 1])

    im2 = axs[1, 0].imshow(mask_vis, cmap=mask_cmap, norm=mask_norm)
    axs[1, 0].set_title("RGB Mask (Transparent=Red)")
    axs[1, 0].axis("off")
    fig.colorbar(im2, ax=axs[1, 0], ticks=[0, 1])

    im3 = axs[1, 1].imshow(depth_gt_vis, cmap='jet')
    axs[1, 1].set_title("Depth Ground Truth")
    axs[1, 1].axis("off")
    fig.colorbar(im3, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()

# def light_plot_image_process(depth, depth_gt, rgb, rgb_mask):
    
#     rgb_copy = rgb.copy()
#     depth_copy = depth.copy()
#     rgb_mask_copy = rgb_mask.copy()
#     ori_depth = depth
    
    
#     depth_gt[np.isnan(depth_gt)] = 0.0
#     # depth_gt[rgb_mask != 0] = 0
#     rgb_mask = np.where(depth_gt < 0.000000001, 255, 0).astype(np.uint8)
#     depth = cv2.resize(depth, (320, 240), interpolation = cv2.INTER_NEAREST)
#     depth_copy = cv2.resize(depth_copy, (320, 240), interpolation = cv2.INTER_NEAREST)
#     depth_gt = cv2.resize(depth_gt, (320, 240), interpolation = cv2.INTER_NEAREST)
#     rgb_mask = cv2.resize(rgb_mask, (320, 240), interpolation = cv2.INTER_NEAREST).astype(np.uint8)
#     rgb_mask_copy = cv2.resize(rgb_mask_copy, (320, 240), interpolation = cv2.INTER_NEAREST).astype(np.uint8)
#     rgb_mask_copy[rgb_mask_copy != 0] = 1
#     rgb_mask[rgb_mask != 0] = 1

#     depth_mask = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
#     depth_mask[depth_mask != 0] = 1
#     # depth_gt_new = handle_depth(depth_gt.copy(), depth_gt.copy(), rgb_mask_copy)
    
#     neg_zero_mask = np.where(depth_gt < 0.0000001)
    
    
#     depth_gt[neg_zero_mask] = 0
#     depth[neg_zero_mask] = 0
#     neg_zero_mask = np.where(depth_gt > 5)
#     depth_gt[neg_zero_mask] = 0
#     depth[neg_zero_mask] = 0

#     depth = np.clip(depth, 0, 1)
#     depth_gt = np.clip(depth_gt, 0, 1)


#     fig, axs = plt.subplots(2, 2)
#     tt="jet"
#     rgb_1=rgb_copy
#     rgb_1 = cv2.resize(rgb_1,(320,240))
#     ori_depth = cv2.resize(ori_depth,(320,240))

#     ori_depth = depth*255
#     ori_depth= ori_depth.astype(int)
#     depth_gt= depth_gt*255
#     depth_gt= depth_gt.astype(int)

#     axs.flat[0].imshow(rgb_1)
#     axs.flat[0].set_title("rgb")

#     axs.flat[1].imshow(ori_depth,cmap=tt)
#     axs.flat[1].set_title("original")

#     axs.flat[2].imshow(rgb_mask.astype(np.float32) * 255, cmap='jet')
#     axs.flat[2].set_title("rgb_mask")

#     axs.flat[3].imshow(depth_gt,cmap=tt)
#     axs.flat[3].set_title("groud truth")


#     plt.show()
    

def plot_realat_depth(depth):

    
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().squeeze().numpy()

    # Prevent extreme values from interfering with visualization
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0, 255]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    # Apply colormap
    cmap = matplotlib.colormaps.get_cmap('jet')  # or 'plasma', 'turbo'
    # cmap = plt.get_cmap('jet')
    depth_colored = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

    # Show image
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_colored)
    plt.title('title')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_image_process(depth, depth_gt, res, rgb, rgb_mask, rgb_mask_ori):
    # Copy inputs to avoid modifying originals
    rgb_copy = rgb.copy()
    depth_copy = depth.copy()
    ori_depth = depth.copy()

    # Handle invalid values
    depth_gt[np.isnan(depth_gt)] = 0.0
    rgb_mask = np.where(depth_gt < 1e-8, 255, 0).astype(np.uint8)

    # Resize all to same size
    target_size = (320, 240)
    rgb_1 = cv2.resize(rgb_copy, target_size)
    depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
    depth_gt = cv2.resize(depth_gt, target_size, interpolation=cv2.INTER_NEAREST)
    res = cv2.resize(res, target_size, interpolation=cv2.INTER_NEAREST)
    rgb_mask = cv2.resize(rgb_mask, target_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    rgb_mask_ori = cv2.resize(rgb_mask_ori, target_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # Clean mask
    rgb_mask[rgb_mask != 0] = 1
    rgb_mask_ori[rgb_mask_ori != 0] = 1

    # Filter invalid or far values
    invalid_mask = (depth_gt < 1e-7) | (depth_gt > 5)
    depth[invalid_mask] = 0
    depth_gt[invalid_mask] = 0
    res[invalid_mask] = 0

    # Normalize for display
    res_vis = np.clip(res, 0, 1) * 255
    depth_vis = np.clip(depth, 0, 1) * 255
    depth_gt_vis = np.clip(depth_gt, 0, 1) * 255

    res_vis = res_vis.astype(np.uint8)
    depth_vis = depth_vis.astype(np.uint8)
    depth_gt_vis = depth_gt_vis.astype(np.uint8)

    # Compute masked error
    # eps = 1e-5
    # error = rgb_mask_ori * np.abs(res_vis - depth_gt_vis) / (depth_gt_vis + eps)

    error = rgb_mask_ori*abs(res-depth_gt)/(depth_gt+0.00001)
#     # error = rgb_mask_ori*abs(res-depth_gt)/255
#     # error = res-depth_gt

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    cmap = 'jet'

    axs[0, 0].imshow(rgb_1)
    axs[0, 0].set_title("RGB")

    axs[0, 1].imshow(depth_vis, cmap=cmap)
    axs[0, 1].set_title("Original Depth")

    axs[0, 2].imshow(res_vis, cmap=cmap)
    axs[0, 2].set_title("Model Output")

    axs[1, 0].imshow(depth_gt_vis, cmap=cmap)
    axs[1, 0].set_title("Ground Truth")

    axs[1, 1].imshow(rgb_mask_ori, cmap='gray')
    axs[1, 1].set_title("Mask")

    im = axs[1, 2].imshow(error, vmin=0, vmax=0.2, cmap=cmap)
    axs[1, 2].set_title("Masked Error")
    fig.colorbar(im, ax=axs[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
# def plot_image_process(depth, depth_gt, res, rgb, rgb_mask, rgb_mask_ori):
    
#     rgb_copy = rgb.copy()
#     depth_copy = depth.copy()
#     rgb_mask_copy = rgb_mask.copy()
#     ori_depth = depth
    
    
#     depth_gt[np.isnan(depth_gt)] = 0.0
#     # depth_gt[rgb_mask != 0] = 0
#     rgb_mask = np.where(depth_gt < 0.000000001, 255, 0).astype(np.uint8)
#     depth = cv2.resize(depth, (320, 240), interpolation = cv2.INTER_NEAREST)
#     depth_copy = cv2.resize(depth_copy, (320, 240), interpolation = cv2.INTER_NEAREST)
#     depth_gt = cv2.resize(depth_gt, (320, 240), interpolation = cv2.INTER_NEAREST)
#     res = cv2.resize(res, (320, 240), interpolation = cv2.INTER_NEAREST)
#     rgb_mask = cv2.resize(rgb_mask, (320, 240), interpolation = cv2.INTER_NEAREST).astype(np.uint8)
#     rgb_mask_copy = cv2.resize(rgb_mask_copy, (320, 240), interpolation = cv2.INTER_NEAREST).astype(np.uint8)
#     rgb_mask_copy[rgb_mask_copy != 0] = 1
#     rgb_mask[rgb_mask != 0] = 1

#     depth_mask = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
#     depth_mask[depth_mask != 0] = 1
#     # depth_gt_new = handle_depth(depth_gt.copy(), depth_gt.copy(), rgb_mask_copy)
    
#     neg_zero_mask = np.where(depth_gt < 0.0000001)
    
    
#     res[neg_zero_mask] = 0
#     depth_gt[neg_zero_mask] = 0
#     depth[neg_zero_mask] = 0
#     neg_zero_mask = np.where(depth_gt > 5)
#     res[neg_zero_mask] = 0
#     depth_gt[neg_zero_mask] = 0
#     depth[neg_zero_mask] = 0

#     res = np.clip(res, 0, 1)
#     depth = np.clip(depth, 0, 1)
#     depth_gt = np.clip(depth_gt, 0, 1)


#     fig, axs = plt.subplots(2, 3)
#     tt="jet"
#     rgb_1=rgb_copy
#     rgb_1 = cv2.resize(rgb_1,(320,240))
#     ori_depth = cv2.resize(ori_depth,(320,240))

#     res = res*255
#     res = res.astype(int)
#     ori_depth = depth*255
#     ori_depth= ori_depth.astype(int)
#     depth_gt= depth_gt*255
#     depth_gt= depth_gt.astype(int)

#     axs.flat[0].imshow(rgb_1)
#     axs.flat[0].set_title("rgb")

#     axs.flat[1].imshow(ori_depth,cmap=tt)

#     axs.flat[1].set_title("original")

#     axs.flat[2].imshow(res,cmap=tt)
#     axs.flat[2].set_title("model output")

#     axs.flat[3].imshow(depth_gt,cmap=tt)
#     axs.flat[3].set_title("groud truth")

#     axs.flat[4].imshow(rgb_mask_ori)
#     axs.flat[4].set_title("mask")

#     error = rgb_mask_ori*abs(res-depth_gt)/(depth_gt+0.00001)
#     # error = rgb_mask_ori*abs(res-depth_gt)/255
#     # error = res-depth_gt

#     axs.flat[5].imshow(error,vmax=0.2,vmin=0, cmap=tt)
#     axs.flat[5].set_title("masked_res")

#     sc = plt.imshow(error, vmax=0.2, vmin=0,cmap = plt.cm.jet)# 限制范围为0-100
#     plt.colorbar()
#     # plt.savefig("transcg_sence21_1.png")

#     plt.show()
    
    
def depth_to_point_cloud(depth_map, color_image, intrinsics):
    """depth image to points"""
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            d = depth_map[v, u]
            if 0.2 < d < 1.0:  # 忽略无效深度值
                z = d
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def depth_to_point_cloud_no_color(depth_gt, intrinsics, depth=None, depth_mask=None, seprate=False, epsilon=5e-4):
    """depth image to points without color"""
    h, w = depth_gt.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    if seprate:
        trans_mask = (depth_mask == 1)

        # region mask
        reflect_mask = (depth == 0) & trans_mask
        normal_mask = (np.abs(depth - depth_gt) < epsilon) & trans_mask
        refract_mask = ~(reflect_mask | normal_mask) & trans_mask
    
    points = []
    colors = []
    color = [1.0, 0.45, 0.45]
    for v in range(h):
        for u in range(w):
            d = depth_gt[v, u]
            if 0.2 < d < 1.0:  # ignore invalid depth values
                z = d
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                if seprate:
                    if refract_mask[v, u]:
                        color = [1.0, 0.45, 0.45]  # light red
                    elif reflect_mask[v, u]:
                        color = [0.6, 0.8, 1.0]    # light blue
                    elif normal_mask[v, u]:
                        color = [0.7, 1.0, 0.7]    # light green
                    else:
                        continue  # ignore unclassified pixels

                    colors.append(color) # light red
                else:
                    colors.append(color) # light red

                points.append([x, y, z])
                
                
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def real_world_depth_to_point_cloud(depth_map, color_image, intrinsics):
    """depth image to point cloud"""
    h, w = depth_map.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            d = depth_map[v, u]
            if 0.2 < d < 1.0:  # ignore invalid depth values
                z = d
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd


def pred_real_world_depth_to_point_cloud(depth_map, color_image, intrinsics):
    """depth image to point cloud"""
    h, w = depth_map.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            d = depth_map[v, u]
            if 0.2 < d < 1.0:  # ignore invalid depth values
                z = d
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                colors.append(color_image[v, u] / 255.0)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    num_points = 50  # Minimum number of neighbors within radius
    radius = 0.02     # Neighborhood radius

    # Remove outliers based on radius
    pcd, ind = pcd.remove_radius_outlier(num_points, radius)

    # Return filtered Open3D point cloud object
    return pcd


def vis_points(pcd):
    
    # while True:
    o3d.visualization.draw_geometries([pcd])
    # key = cv2.waitKey(1)

        # if key & 0xFF == ord('q') or key == 27:
        #     break


def analyze_transparent_depth_error(
    rgb, depth, depth_gt, depth_mask, pred_depth,
    error_thresh=0.03, epsilon=1e-4, cmap='viridis'
):
    """
    Visualize and analyze depth prediction errors in transparent regions.

    Parameters:
    - rgb: (H, W, 3) color image
    - depth: (H, W) original input depth map (including refraction and reflection)
    - depth_gt: (H, W) GT depth map
    - depth_mask: (H, W) binary mask for transparent regions (1 indicates transparent)
    - pred_depth: (H, W) predicted depth map
    - error_thresh: upper limit for filtering outliers in error heatmap
    - epsilon: tolerance for determining "normal" regions
    - cmap: colormap for error heatmap
    """

    abs_error = np.abs(pred_depth - depth_gt)
    trans_mask = (depth_mask == 1)

    # Three types of region masks
    reflect_mask = (depth == 0) & trans_mask
    normal_mask = (np.abs(depth - depth_gt) < epsilon) & trans_mask
    refract_mask = ~(reflect_mask | normal_mask) & trans_mask

    # Region error extraction (while removing outliers)
    refract_error = abs_error[refract_mask & (abs_error < error_thresh)]
    reflect_error = abs_error[reflect_mask & (abs_error < error_thresh)]
    normal_error  = abs_error[normal_mask  & (abs_error < error_thresh)]

    # Region size statistics (only for transparent areas)
    total_trans_pixels = np.sum(trans_mask)
    reflect_ratio = np.sum(reflect_mask) / total_trans_pixels if total_trans_pixels else 0
    refract_ratio = np.sum(refract_mask) / total_trans_pixels if total_trans_pixels else 0
    normal_ratio = np.sum(normal_mask) / total_trans_pixels if total_trans_pixels else 0

    # print(refract_error.shape, reflect_error.shape, normal_error.shape)

    # -------- Visualization Preparation --------
    # RGB Overlay Region Colors (Red=Refracted, Blue=Reflected)
    # overlay = np.zeros_like(rgb, dtype=np.uint8)
    # overlay[reflect_mask] = [0, 0, 255]
    # overlay[refract_mask] = [255, 0, 0]
    # blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    # Create a black image as the overlay layer
    overlay = np.zeros_like(rgb, dtype=np.uint8)

    # Assign colors only in transparent regions (Blue = Reflected, Red = Refracted)
    overlay[reflect_mask] = [0, 0, 255]
    overlay[refract_mask] = [255, 0, 0]

    # In the original RGB image, set non-transparent areas to black
    rgb_masked = rgb.copy()
    rgb_masked[depth_mask != 1] = [255, 255, 255]

    # Blend the overlay layer
    # blended = cv2.addWeighted(rgb_masked, 0.6, overlay, 0.4, 0)
    # Initialize blended image
    blended = rgb_masked.copy()

    # Apply weighted blending only in transparent areas (to avoid gray background)
    alpha = 0.6
    blended[depth_mask == 1] = (
        alpha * rgb_masked[depth_mask == 1] + (1 - alpha) * overlay[depth_mask == 1]
    ).astype(np.uint8)

    # Error heatmap, only showing transparent areas & errors within reasonable range
    error_vis = np.where(trans_mask & (abs_error < error_thresh), abs_error, np.nan)
    if np.all(np.isnan(error_vis)):
        error_vis = np.zeros_like(abs_error)

    # -------- Visualization Output (Four-in-One)--------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # image 1：RGB + region overlay
    axs[0, 0].imshow(blended)
    axs[0, 0].set_title("Region Overlay\nRed=Refracted, Blue=Reflected")
    axs[0, 0].axis('off')

    # image 2：error heatmap
    im1 = axs[0, 1].imshow(error_vis, cmap=cmap, vmin=0, vmax=error_thresh)
    axs[0, 1].set_title(f"Transparent Region Error Map\n(Error < {error_thresh} m)")
    axs[0, 1].axis('off')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, label='Error (m)')

    # image 3：error heatmap + region boundaries
    im2 = axs[1, 0].imshow(error_vis, cmap=cmap, vmin=0, vmax=error_thresh)
    axs[1, 0].contour(refract_mask, levels=[0.5], colors='red', linewidths=1)
    axs[1, 0].contour(reflect_mask, levels=[0.5], colors='blue', linewidths=1)
    axs[1, 0].set_title("Error Heatmap + Region Boundaries")
    axs[1, 0].axis('off')
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04, label='Error (m)')

    # image 4：error boxplot
    axs[1, 1].boxplot(
        [refract_error, reflect_error, normal_error],
        labels=["Refracted", "Reflected", "Normal"],
        showfliers=False
    )
    axs[1, 1].set_ylabel("Absolute Depth Error (m)")
    axs[1, 1].set_title("Prediction Error by Region")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # -------- Error Statistics Output --------
    def compute_stats(errors):
        if len(errors) == 0:
            return {
                'Count': 0,
                'MAE': 0.0,
                'RMSE': 0.0,
                'Max Error': 0.0,
                'Std': 0.0
            }
        return {
            'Count': len(errors),
            'MAE': np.mean(errors),
            'RMSE': np.sqrt(np.mean(errors ** 2)),
            'Max Error': np.max(errors),
            'Std': np.std(errors)
        }

    stats = {
        'Refracted': compute_stats(refract_error),
        'Reflected': compute_stats(reflect_error),
        'Normal': compute_stats(normal_error)
    }

    df_stats = pd.DataFrame(stats).T
    df_stats['Region Ratio'] = [refract_ratio, reflect_ratio, normal_ratio]
    print("region error statistics:")
    print(df_stats.to_string(float_format="%.4f"))

    return df_stats







def visualize_input_depth_analysis(
    depth, depth_gt, depth_mask, rgb=None,
    epsilon=1e-4,
    cmap='viridis'
):
    """
    Visualize the classification and error heatmap of the input depth map in transparent regions.
    Combine into three subplots and add light red contours to highlight the refracted regions.
    """

    # Ensure inputs are 2D
    depth = np.squeeze(depth)
    depth_gt = np.squeeze(depth_gt)
    depth_mask = np.squeeze(depth_mask)

    assert depth.shape == depth_gt.shape == depth_mask.shape, "Shape mismatch"

    # Region partitioning
    depth_error = np.abs(depth - depth_gt)
    trans_mask = (depth_mask == 1)

    reflect_mask = (depth == 0) & trans_mask
    normal_mask = (depth != 0) & (depth_error < epsilon) & trans_mask
    refract_mask = ~(reflect_mask | normal_mask) & trans_mask

    
    region_map = np.ones((*depth.shape, 3), dtype=np.uint8) * 255
    region_map[refract_mask] = [255, 100, 100]     # Red - Refracted
    region_map[normal_mask] = [180, 255, 180]      # Green - Normal
    region_map[reflect_mask] = [180, 220, 255]      # Blue - Reflected

    # Heatmap Mask (Exclude Reflected Regions)
    error_mask = trans_mask & (depth != 0)
    error_vis = np.where(error_mask, depth_error, np.nan)

    if np.all(np.isnan(error_vis)):
        error_vis = np.zeros_like(depth_error)

    vmax = np.nanpercentile(error_vis, 95)
    vmin = 0

    # ====== create subplots ======
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # image 1：region classification image
    axs[0, 1].imshow(region_map)
    axs[0, 1].set_title("Region Classification\nRed=Refracted, Green=Normal")
    axs[0, 1].axis('off')

    # image 2：error heatmap (exclude reflected)
    im1 = axs[1, 0].imshow(error_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title(f"Depth Error Heatmap\n(Reflected Area Removed)")
    axs[1, 0].axis('off')
    fig.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # image 3：error heatmap + refracted region contours (light red)
    # im2 = axs[2].imshow(error_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    # axs[2].contour(refract_mask, levels=[0.5], colors='lightcoral', linewidths=1.2)
    # axs[2].set_title("Refracted Region Highlighted")
    # axs[2].axis('off')
    # fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    im2 = axs[1, 1].imshow(error_vis, cmap=cmap, vmin=vmin, vmax=vmax)

    # Create light red overlay (only refracted regions visible, others transparent)
    overlay = np.zeros((*refract_mask.shape, 4), dtype=np.float32)  # RGBA
    overlay[..., 0] = 1.0       # Red channel
    overlay[..., 3] = refract_mask.astype(np.float32) * 0.2  # Alpha channel: refracted region transparency

    axs[1, 1].imshow(overlay)
    # axs[2].contour(refract_mask, levels=[0.5], colors='lightcoral', linewidths=1.2)
    axs[1, 1].contour(refract_mask, levels=[0.5], colors='red', linewidths=1.2)

    axs[1, 1].set_title("Refracted Region Masked (Light Red)")
    axs[1, 1].axis('off')
    fig.colorbar(im2, ax=axs[1, 1], fraction=0.046, pad=0.04)

    # mask
    # axs[0, 0].imshow(depth_mask, cmap='gray')
    # axs[0, 0].set_title("Transparent Region Mask")
    # axs[0, 0].axis('off')

    white_bg = np.ones((*depth_mask.shape, 3), dtype=np.float32)
    mask_overlay = np.zeros((*depth_mask.shape, 4), dtype=np.float32)
    mask_overlay[..., :3] = [0.6, 0.8, 1.0]  # Light blue RGB
    mask_overlay[..., 3] = depth_mask.astype(np.float32) * 0.5  # alpha

    axs[0, 0].imshow(white_bg)
    axs[0, 0].imshow(mask_overlay)
    axs[0, 0].set_title("Transparent Region Mask\n(White BG + Light Blue Overlay)")
    axs[0, 0].axis('off')

    plt.tight_layout()
    plt.savefig("output_name.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    return {
        "refract_mask": refract_mask,
        "reflect_mask": reflect_mask,
        "normal_mask": normal_mask,
        "error_map": error_vis
    }


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def run_gradcam_on_encoder_img(model, rgb, depth, relative_depth, mask, target_block=1):
    """
    Visualize the attention block of encoder_img (SwinTransformer).
    default layer4 target_block block.
    """
    model.eval()
    device = next(model.parameters()).device

    input_tensor = torch.cat([rgb, relative_depth, mask], dim=1).to(device)

    swin = model.encoder_img

    target_layer = swin.layers[3].blocks[target_block]  # 可以改成 layers[2].blocks[x] 等

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')

    grayscale_cam = cam(input_tensor=input_tensor)[0]  # 取第一个样本

    rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
    cam_image = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    plt.imshow(cam_image)
    plt.title("Grad-CAM on SwinTransformer Encoder")
    plt.axis("off")
    plt.show()
