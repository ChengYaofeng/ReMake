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
    可视化 RGB 图、原始深度图、Ground Truth 深度图以及 RGB mask。
    输入图像大小可变，内部统一 resize 到 320x240 显示。
    """
    # 拷贝原始数据
    rgb_vis = cv2.resize(rgb.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)
    depth_vis = cv2.resize(depth.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)
    depth_gt_vis = cv2.resize(depth_gt.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)
    mask_vis = cv2.resize(rgb_mask.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)

    # 处理非法值
    depth_gt_vis[np.isnan(depth_gt_vis)] = 0.0
    mask_vis = np.where(depth_gt_vis < 1e-7, 1, 0).astype(np.uint8)  # 透明区域为1

    # 删除不合理深度（如 > 5m）
    depth_vis[depth_gt_vis > 5] = 0
    depth_gt_vis[depth_gt_vis > 5] = 0

    # clip 并归一化后转为 [0, 255] 显示
    depth_vis = np.clip(depth_vis, 0, 1) * 255
    depth_gt_vis = np.clip(depth_gt_vis, 0, 1) * 255

    depth_vis = depth_vis.astype(np.uint8)
    depth_gt_vis = depth_gt_vis.astype(np.uint8)

    # 自定义 mask 显示：1 = 红色，0 = 黑色
    mask_cmap = colors.ListedColormap(['black', 'red'])
    mask_bounds = [0, 0.5, 1]
    mask_norm = colors.BoundaryNorm(mask_bounds, mask_cmap.N)

    # 画图
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

    # 防止极端值干扰可视化
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
    """将深度图转换为点云"""
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
    """将深度图转换为点云"""
    h, w = depth_gt.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    if seprate:
        trans_mask = (depth_mask == 1)

        # 三类区域 mask
        reflect_mask = (depth == 0) & trans_mask
        normal_mask = (np.abs(depth - depth_gt) < epsilon) & trans_mask
        refract_mask = ~(reflect_mask | normal_mask) & trans_mask
    
    points = []
    colors = []
    color = [1.0, 0.45, 0.45]
    for v in range(h):
        for u in range(w):
            d = depth_gt[v, u]
            if 0.2 < d < 1.0:  # 忽略无效深度值
                z = d
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                if seprate:
                    if refract_mask[v, u]:
                        color = [1.0, 0.45, 0.45]  # 浅红
                    elif reflect_mask[v, u]:
                        color = [0.6, 0.8, 1.0]    # 浅蓝
                    elif normal_mask[v, u]:
                        color = [0.7, 1.0, 0.7]    # 浅绿
                    else:
                        continue  # 忽略未分类像素

                    colors.append(color) # light red
                else:
                    colors.append(color) # light red

                points.append([x, y, z])
                
                
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    return pcd

def real_world_depth_to_point_cloud(depth_map, color_image, intrinsics):
    """将深度图转换为点云"""
    h, w = depth_map.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
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


def pred_real_world_depth_to_point_cloud(depth_map, color_image, intrinsics):
    """将深度图转换为点云"""
    h, w = depth_map.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
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

    num_points = 50  # 半径内最小邻居数
    radius = 0.02     # 邻域半径

    # 基于半径的离群点移除
    pcd, ind = pcd.remove_radius_outlier(num_points, radius)

    # 返回经过过滤的 Open3D 点云对象
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
    可视化并分析透明区域中的折射、反射、正常区域的深度预测误差。

    参数:
    - rgb: (H, W, 3) 彩色图像
    - depth: (H, W) 原始输入深度图（含折射和反射）
    - depth_gt: (H, W) GT 深度图
    - depth_mask: (H, W) 透明区域的二值 mask（1 表示透明）
    - pred_depth: (H, W) 预测的深度图
    - error_thresh: 过滤误差热图中的离群值上限
    - epsilon: 用于判断“正常区域”的容差
    - cmap: 误差热图颜色方案
    """

    abs_error = np.abs(pred_depth - depth_gt)
    trans_mask = (depth_mask == 1)

    # 三类区域 mask
    reflect_mask = (depth == 0) & trans_mask
    normal_mask = (np.abs(depth - depth_gt) < epsilon) & trans_mask
    refract_mask = ~(reflect_mask | normal_mask) & trans_mask

    # 区域误差提取（同时去除异常值）
    refract_error = abs_error[refract_mask & (abs_error < error_thresh)]
    reflect_error = abs_error[reflect_mask & (abs_error < error_thresh)]
    normal_error  = abs_error[normal_mask  & (abs_error < error_thresh)]

    # 区域大小统计（只看透明区域）
    total_trans_pixels = np.sum(trans_mask)
    reflect_ratio = np.sum(reflect_mask) / total_trans_pixels if total_trans_pixels else 0
    refract_ratio = np.sum(refract_mask) / total_trans_pixels if total_trans_pixels else 0
    normal_ratio = np.sum(normal_mask) / total_trans_pixels if total_trans_pixels else 0

    # print(refract_error.shape, reflect_error.shape, normal_error.shape)

    # -------- 可视化准备 --------
    # RGB 叠加区域颜色（红=折射，蓝=反射）
    # overlay = np.zeros_like(rgb, dtype=np.uint8)
    # overlay[reflect_mask] = [0, 0, 255]
    # overlay[refract_mask] = [255, 0, 0]
    # blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    # 创建全黑图像作为 overlay 叠加层
    overlay = np.zeros_like(rgb, dtype=np.uint8)

    # 仅在透明区域中赋予颜色（蓝色 = 反射，红色 = 折射）
    overlay[reflect_mask] = [0, 0, 255]
    overlay[refract_mask] = [255, 0, 0]

    # 原始 RGB 图中，非透明区域设为黑色
    rgb_masked = rgb.copy()
    rgb_masked[depth_mask != 1] = [255, 255, 255]

    # 混合叠加图层
    # blended = cv2.addWeighted(rgb_masked, 0.6, overlay, 0.4, 0)
    # 初始化混合图像
    blended = rgb_masked.copy()

    # 只对透明区域执行加权混合（避免灰色背景）
    alpha = 0.6
    blended[depth_mask == 1] = (
        alpha * rgb_masked[depth_mask == 1] + (1 - alpha) * overlay[depth_mask == 1]
    ).astype(np.uint8)

    # 误差热图，仅显示透明区域 & 误差在合理范围内
    error_vis = np.where(trans_mask & (abs_error < error_thresh), abs_error, np.nan)
    if np.all(np.isnan(error_vis)):
        error_vis = np.zeros_like(abs_error)

    # -------- 可视化输出（四合一）--------
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 图 1：RGB + 区域叠加
    axs[0, 0].imshow(blended)
    axs[0, 0].set_title("Region Overlay\nRed=Refracted, Blue=Reflected")
    axs[0, 0].axis('off')

    # 图 2：透明区域误差热图
    im1 = axs[0, 1].imshow(error_vis, cmap=cmap, vmin=0, vmax=error_thresh)
    axs[0, 1].set_title(f"Transparent Region Error Map\n(Error < {error_thresh} m)")
    axs[0, 1].axis('off')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04, label='Error (m)')

    # 图 3：误差热图 + 区域边界
    im2 = axs[1, 0].imshow(error_vis, cmap=cmap, vmin=0, vmax=error_thresh)
    axs[1, 0].contour(refract_mask, levels=[0.5], colors='red', linewidths=1)
    axs[1, 0].contour(reflect_mask, levels=[0.5], colors='blue', linewidths=1)
    axs[1, 0].set_title("Error Heatmap + Region Boundaries")
    axs[1, 0].axis('off')
    fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04, label='Error (m)')

    # 图 4：误差箱线图
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

    # -------- 误差统计输出 --------
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
    print("区域误差统计：")
    print(df_stats.to_string(float_format="%.4f"))

    return df_stats







def visualize_input_depth_analysis(
    depth, depth_gt, depth_mask, rgb=None,
    epsilon=1e-4,
    cmap='viridis'
):
    """
    可视化输入深度图在透明区域中反射 / 折射 / 正常区域分类与误差热图。
    合并为三个子图显示，并添加浅红色轮廓标注折射区域。
    """

    # 确保输入为2D
    depth = np.squeeze(depth)
    depth_gt = np.squeeze(depth_gt)
    depth_mask = np.squeeze(depth_mask)

    assert depth.shape == depth_gt.shape == depth_mask.shape, "Shape mismatch"

    # 区域划分
    depth_error = np.abs(depth - depth_gt)
    trans_mask = (depth_mask == 1)

    reflect_mask = (depth == 0) & trans_mask
    normal_mask = (depth != 0) & (depth_error < epsilon) & trans_mask
    refract_mask = ~(reflect_mask | normal_mask) & trans_mask

    # RGB 分类可视化图：白底 + 区域染色
    region_map = np.ones((*depth.shape, 3), dtype=np.uint8) * 255
    region_map[refract_mask] = [255, 100, 100]     # 红 - 折射
    region_map[normal_mask] = [180, 255, 180]      # 绿 - 正常
    region_map[reflect_mask] = [180, 220, 255]      # 蓝 - 正常


    # 热图遮罩（排除反射区域）
    error_mask = trans_mask & (depth != 0)
    error_vis = np.where(error_mask, depth_error, np.nan)

    if np.all(np.isnan(error_vis)):
        error_vis = np.zeros_like(depth_error)

    vmax = np.nanpercentile(error_vis, 95)
    vmin = 0

    # ====== 创建三个子图 ======
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 图1：区域分类图
    axs[0, 1].imshow(region_map)
    axs[0, 1].set_title("Region Classification\nRed=Refracted, Green=Normal")
    axs[0, 1].axis('off')

    # 图2：误差热图（排除反射）
    im1 = axs[1, 0].imshow(error_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title(f"Depth Error Heatmap\n(Reflected Area Removed)")
    axs[1, 0].axis('off')
    fig.colorbar(im1, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # 图3：误差热图 + 折射区域轮廓（浅红色）
    # im2 = axs[2].imshow(error_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    # axs[2].contour(refract_mask, levels=[0.5], colors='lightcoral', linewidths=1.2)
    # axs[2].set_title("Refracted Region Highlighted")
    # axs[2].axis('off')
    # fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    im2 = axs[1, 1].imshow(error_vis, cmap=cmap, vmin=vmin, vmax=vmax)

    # 创建浅红色蒙版（仅折射区域可见，其他为透明）
    overlay = np.zeros((*refract_mask.shape, 4), dtype=np.float32)  # RGBA
    overlay[..., 0] = 1.0       # 红色通道
    overlay[..., 3] = refract_mask.astype(np.float32) * 0.2  # Alpha 通道：折射区域透明度

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
    mask_overlay[..., :3] = [0.6, 0.8, 1.0]  # 淡蓝色 RGB
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
    可视化 encoder_img (SwinTransformer) 的 attention block。
    默认可视化 layer4 的第 target_block 个 block。
    """
    model.eval()
    device = next(model.parameters()).device

    # 准备输入：RGB + 相对深度 + mask （[B, 5, H, W]）
    input_tensor = torch.cat([rgb, relative_depth, mask], dim=1).to(device)

    # 手动 forward encoder_img 提取中间层
    swin = model.encoder_img

    # 设定 target layer：如 layers[3] 是最后一层，blocks[x] 是其中第 x 个 block
    target_layer = swin.layers[3].blocks[target_block]  # 可以改成 layers[2].blocks[x] 等

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')

    # 生成 Grad-CAM mask
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # 取第一个样本

    # 可视化
    rgb_np = rgb[0].permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
    cam_image = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    plt.imshow(cam_image)
    plt.title("Grad-CAM on SwinTransformer Encoder")
    plt.axis("off")
    plt.show()
