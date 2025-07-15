import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from run_utils.inferencer import Inferencer
import pyrealsense2 as rs
import time

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
    points_x = (xmap - cx) / fx * points_z
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
        
        return color_image_rgb, depth_image, color_image
    
    def release(self):
        self.pipeline.stop()

def live_inference(args):
    # Initialize camera
    rs_camera = RealSenseCamera()
    
    # Initialize model
    inferencer = Inferencer(args)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window("RealSense Point Cloud", width=1280, height=720)
    
    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    vis.add_geometry(frame)
    
    # Add reference sphere at specific depth
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
    vis.add_geometry(sphere)
    
    # Create initial point clouds
    pcd_pred = o3d.geometry.PointCloud()
    pcd_raw = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_pred)
    vis.add_geometry(pcd_raw)
    
    try:
        while True:
            # Get frames from camera
            rgb, depth, color_display = rs_camera.get_frames()
            # print(rgb.shape, depth.shape)
            # Run inference
            # depth = np.where((depth >= 0.2) & (depth <= 1.0), depth, 0.0)
            # depth = np.clip(depth, 0.0, 0.8)
            depth[np.isnan(depth)] = 0.0
            depth = np.where(depth < 0., 0, depth)
            depth = np.where(depth > 1.5, 1.5, depth)
            res = inferencer.inference(rgb, depth)
            
            # Clip depths for visualization
            # res = np.clip(res, 0.3, 1.0)
            # depth = np.clip(depth, 0.3, 1.0)
            
            # Create point clouds
            cloud_pred = draw_point_cloud(rgb, res, rs_camera.intrinsics, scale=1.0)
            # cloud_raw = draw_point_cloud(rgb, depth, rs_camera.intrinsics, scale=1.0)
            
            # Update point clouds
            pcd_pred.points = cloud_pred.points
            pcd_pred.colors = cloud_pred.colors
            # pcd_raw.points = cloud_raw.points
            # pcd_raw.colors = cloud_raw.colors
            
            # Update visualization
            vis.update_geometry(pcd_pred)
            # vis.update_geometry(pcd_raw)
            
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            # Display RGB and depth images
            cv2.imshow("RGB", color_display)
            depth_display = cv2.applyColorMap(cv2.convertScaleAbs(depth*255, alpha=1), cv2.COLORMAP_JET)
            cv2.imshow("Raw Depth", depth_display)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
            elif key & 0xFF == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                cv2.imwrite(f"rgb_{timestamp}.png", color_display)
                np.save(f"depth_{timestamp}.npy", depth)
                print(f"Saved frame at timestamp {timestamp}")
                
    finally:
        # Clean up
        rs_camera.release()
        vis.destroy_window()
        cv2.destroyAllWindows()









