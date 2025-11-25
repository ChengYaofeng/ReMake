import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from run_utils.inferencer import Inferencer
import pyrealsense2 as rs
import time
from utils.visualization import vis_points, depth_to_point_cloud, real_world_depth_to_point_cloud, pred_real_world_depth_to_point_cloud
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def draw_point_cloud(color, depth, camera_intrinsics, use_mask=False, use_inpainting=True, scale=1.0, inpainting_radius=5, fault_depth_limit=0.2, epsilon=0.01):
    """
    Given the depth image, return the point cloud in open3d format.
    The code is adapted from [graspnet.py] in the [graspnetAPI] repository.
    """
    d = depth.copy()
    c = color.copy() / 255.0
    # print(d.shape)
    # print(c.shape)
    
    if use_inpainting:
        # Make sure depth is in the right format for inpainting
        fault_mask = (d < fault_depth_limit * scale)
        d[fault_mask] = 0
        
        # Convert to the proper format for inpainting (float32)
        d_inpaint = d.astype(np.float32)
        inpainting_mask = (np.abs(d) < epsilon * scale).astype(np.uint8)
        # print('-'*20)
        
        # Check if mask has any points to inpaint
        if np.sum(inpainting_mask) > 0:
            # print('-'*20)
            d_inpaint = cv2.inpaint(d_inpaint, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)
            d = d_inpaint
    
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    
    points_z = d / scale
    points_x = -(xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    
    if use_mask:
        mask = (points_z > 0)
        points = points[mask]
        c = c[mask]
    else:
        points = points.reshape((-1, 3))
        c = c.reshape((-1, 3))
        
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(c)
    return cloud

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable color and depth streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        self.depth_profile = self.profile.get_stream(rs.stream.depth)
        self.color_profile = self.profile.get_stream(rs.stream.color)
        self.depth_intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsics()
        
        # Create alignment object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
        
        # Get depth scale
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        # Get intrinsics as numpy array
        self.intrinsics = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Apply warmup frames
        for _ in range(30):
            self.pipeline.wait_for_frames()
    
    def get_frames(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert depth to meters
        depth_image = depth_image * self.depth_scale
        
        # Convert BGR to RGB for processing
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        return color_image_rgb, depth_image#, color_image
    
    def release(self):
        self.pipeline.stop()


def realworld_inference(args, **kwargs):

    sam_checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
    sam.to("cuda")  # or "cpu"

    mask_generator = SamAutomaticMaskGenerator(sam)
    
    inferencer = Inferencer(args)

    align = rs.align(rs.stream.color) #depth image align to color image

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) #640, 480,
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # Depth filters
            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)

            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()) / 1000.0
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # cv2.imshow('color image', rgb_bgr)
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)   # float32 → 0-255
            depth_u8   = depth_norm.astype(np.uint8)                           # → uint8
            depth_vis  = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
            vis = np.hstack((rgb_bgr, depth_vis))   
            # cv2.imshow('RGB | Depth', vis)
            scale = 0.6
            vis_resized = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # visualize image
            cv2.imshow('RGB | Depth', vis_resized)

            key = cv2.waitKey(1)
            if key & 0xFF == ord(' '):  # press space to capture and segment
                masks = mask_generator.generate(rgb)
                
                overlay = rgb.copy()
                for i, mask_dict in enumerate(masks[:36]):  # show up to 36
                    mask = mask_dict["segmentation"].astype(np.uint8)
                    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                    overlay[mask > 0] = overlay[mask > 0] * 0.5 + color * 0.5

                    x, y, w, h = mask_dict["bbox"]
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color.tolist(), 2)

                    # show number：0-9, a-z
                    if i < 10:
                        label = str(i)
                    else:
                        label = chr(ord('a') + i - 10)

                    cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

                cv2.imshow("SAM Everything Segmentation", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(0)

                if ord('0') <= key <= ord('9'):
                    selected_idx = key - ord('0')
                elif ord('a') <= key <= ord('z'):
                    selected_idx = 10 + (key - ord('a'))
                else:
                    raise Exception("Please press key 0~9 or a~z")

                if selected_idx >= len(masks):
                    raise Exception(f"Selected index {selected_idx} exceeds available masks ({len(masks)}).")
                
                # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                
                # no_mask_depth = kwargs.get('no_mask_depth', False)
                
                # crop the depth to useful distance
                # depth = np.where((depth <= 1.0), depth, 0.0)
                # depth = np.clip(depth, 0.0, 1.2)
                # depth[np.isnan(depth)] = 0.0
                # depth = np.where(depth < 0., 0, depth)
                # depth = np.where(depth > 10., 0, depth)
                # mask = np.stack(mask, axis=0)
                # print(masks)
                select_mask = masks[selected_idx]['segmentation'].astype(np.uint8)
                # print(select_mask.shape)
                
                kernel = np.ones((3, 3), np.uint8)
                mask_resized = cv2.resize(select_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)  # mask 1
                mask_resized = cv2.erode(select_mask, kernel, iterations=1)
                mask_inv = 1 - mask_resized  # usful mask 0
                
                depth_no_trans = depth * mask_inv
                
                # ablation
                # if no_mask_depth is True:
                    # depth = depth * (1-rgb_mask)
                # print('no mask depth')
                # res = inferencer.inference(rgb, depth_no_trans, mask_resized)
                # else:
                #     res = inferencer.inference(rgb, depth)
                    
                res = inferencer.inference(rgb, depth, mask_resized)

                # visulation pred
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

                depth_8u = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_VIRIDIS)

                cv2.imshow('Predicted Depth', depth_colormap)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # predicted transparent object depth
                pred_mask_depth = mask_resized * res
                # predicted transparent object depth + original depth for non-transparent area
                pred_vis_depth = pred_mask_depth + depth_no_trans

                # visualize original depth
                pcd_gt = real_world_depth_to_point_cloud(depth, rgb, intr)
                vis_points(pcd_gt)
                
                # 
                rgb_mask = np.stack([mask_resized]*3, axis=-1)
                red_layer = np.zeros_like(rgb)
                red_layer[:, :] = [0, 0, 255]
                
                rgb_copy = rgb.copy()
                rgb_copy = np.where(rgb_mask == 1, rgb * 0.5 + red_layer * 0.5, rgb).astype(np.uint8)

                # visualize predicted depth + original depth
                pred_pc = real_world_depth_to_point_cloud(pred_vis_depth, rgb_copy, intr)
                # pred_pc = pred_real_world_depth_to_point_cloud(pred_vis_depth, rgb_copy, intr)

                vis_points(pred_pc)
                # global predicted depth visualization
                # pred_pc = real_world_depth_to_point_cloud(res, rgb_copy, intr)
                # vis_points(pred_pc)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
