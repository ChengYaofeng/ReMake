import os
import numpy as np
import cv2
from run_utils.inferencer import Inferencer
from PIL import Image
from utils.visualization import vis_points, plot_image_process, depth_to_point_cloud, depth_to_point_cloud_no_color, plot_realat_depth, analyze_transparent_depth_error, visualize_input_depth_analysis
import open3d as o3d


# def load_image(idx, jdx):
#     rgb = cv2.imread(f'/home/cyf/YFTrans/datasets/transcg/transcg/scene{idx}/{jdx}/rgb1.png')
#     rgb_mask = np.array(Image.open(f'/home/cyf/YFTrans/datasets/transcg/transcg/scene{idx}/{jdx}/depth1-gt-mask.png'), dtype = np.float32)
#     depth = np.array(Image.open(f'/home/cyf/YFTrans/datasets/transcg/transcg/scene{idx}/{jdx}/depth1.png'), dtype = np.float32)
#     depth_gt = np.array(Image.open(f'/home/cyf/YFTrans/datasets/transcg/transcg/scene{idx}/{jdx}/depth1-gt.png'), dtype = np.float32)
#     rgb_mask_ori = cv2.resize(rgb_mask.copy(), (320, 240), interpolation = cv2.INTER_NEAREST)
    
#     depth = depth / 1000
#     depth_gt = depth_gt / 1000
    
#     return rgb, depth, rgb_mask, rgb_mask_ori, depth_gt
def load_image(idx, jdx):
    base_path = f'/home/cyf/YFTrans/datasets/transcg/transcg/scene{idx}/{jdx}'
    
    try:
        # 构造文件路径
        rgb_path       = os.path.join(base_path, 'rgb1.png')
        mask_path      = os.path.join(base_path, 'depth1-gt-mask.png')
        depth_path     = os.path.join(base_path, 'depth1.png')
        depth_gt_path  = os.path.join(base_path, 'depth1-gt.png')

        # 检查文件是否存在
        for p in [rgb_path, mask_path, depth_path, depth_gt_path]:
            if not os.path.exists(p):
                print(f"[跳过] 文件不存在: {p}")
                return None

        # 加载图像
        rgb         = cv2.imread(rgb_path)
        rgb_mask    = np.array(Image.open(mask_path), dtype=np.float32)
        depth       = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        depth_gt    = np.array(Image.open(depth_gt_path), dtype=np.float32) / 1000.0
        rgb_mask_ori = cv2.resize(rgb_mask.copy(), (320, 240), interpolation=cv2.INTER_NEAREST)

        return rgb, depth, rgb_mask, rgb_mask_ori, depth_gt

    except Exception as e:
        print(f"[错误] 加载 scene{idx}/{jdx} 时发生错误: {e}")
        return None


def inference(args, **kwargs):
    
    inferencer = Inferencer(args)
    
    for i in range(100):
        if i == 0:
            i = 1
        print(i*4)

        # rgb, depth, rgb_mask, rgb_mask_ori, depth_gt = load_image(1, i*5)
        result = load_image(i * 30, 6)
        if result is None:
            continue  # 跳过加载失败的图片
        rgb, depth, rgb_mask, rgb_mask_ori, depth_gt = result
        
        no_mask_depth = kwargs.get('no_mask_depth', False)
        if no_mask_depth is True:
            depth = depth * (1-rgb_mask)
            print('-------------no mask depth--------')
            
        res = inferencer.inference(rgb, depth, rgb_mask)
        # res = res * 255 #这里乘255的作用
        
        # cv2.imwrite('transcg_28_20_TDC_pred.png', res)
        cam_intrinsics = np.load('/home/cyf/YFTrans/datasets/transcg/transcg/camera_intrinsics/1-camIntrinsics-D435.npy')

        # print(depth)
        # print(depth.shape)

        plot_image_process(depth, depth_gt, res, rgb, rgb_mask, rgb_mask_ori)

        analyze_transparent_depth_error(rgb, depth, depth_gt, rgb_mask, res)
        
        visualize_input_depth_analysis(depth, depth_gt, rgb_mask, rgb)

        # plot_realat_depth(depth, depth_gt, res, rgb, depth_rel)

        # masked points
        # depth = depth * rgb_mask
        
        # only mask vis
        # res = res * rgb_mask
        # depth_gt = depth_gt * rgb_mask

        # only mask res
        res = res * rgb_mask
        res = res + depth_gt * (1 - rgb_mask)

        pcd_gt = depth_to_point_cloud(depth_gt, rgb, cam_intrinsics)
        vis_points(pcd_gt)
        
        # 真值红色，预测彩色
        pcd = depth_to_point_cloud(res, rgb, cam_intrinsics)
        vis_points(pcd)

        # # pcd_gt = depth_to_point_cloud(depth_gt, rgb, cam_intrinsics)

        # pcd_gt = depth_to_point_cloud_no_color(depth_gt, cam_intrinsics, depth, rgb_mask, seprate=False)
        # vis_points(pcd_gt)

        
        # o3d.visualization.draw_geometries([pcd_gt, pcd])

        # rgb_relat = cv2.resize(rgb_relat, target_size, interpolation=cv2.INTER_NEAREST)
        

    

if __name__ == '__main__':
    inference()