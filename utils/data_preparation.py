
import cv2
import random
import torch
import numpy as np
from .tools import get_surface_normal_from_depth
import OpenEXR
import Imath
from utils.visualization import depth_to_point_cloud, depth_to_point_cloud_no_color, plot_realat_depth, plot_image_process, light_plot_image_process
import open3d as o3d
from torchvision.transforms import Compose
from utils.transform import Resize, NormalizeImage, PrepareForNet
from time import perf_counter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

DILATION_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)

def plot_rgb_image(img_tensor_or_array, title=None, save_path=None):
    """
    显示 RGB 图像，支持 PyTorch tensor 或 NumPy array
    输入格式可以是：
      - torch.Tensor: (3, H, W) 或 (1, 3, H, W)
      - np.ndarray:  (H, W, 3)

    参数:
        img_tensor_or_array: 输入图像 (Tensor 或 ndarray)
        title: 图像标题（可选）
        save_path: 若设置此路径，则保存图像而不是显示（可选）
    """
    # 如果是 PyTorch tensor
    if isinstance(img_tensor_or_array, torch.Tensor):
        img = img_tensor_or_array.detach().cpu()

        # 去掉 batch 维
        if img.ndim == 4:
            img = img.squeeze(0)

        # 转换为 (H, W, 3)
        img = img.permute(1, 2, 0).numpy()

    elif isinstance(img_tensor_or_array, np.ndarray):
        img = img_tensor_or_array
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    # Clip 到 [0,1] 以防溢出
    img = np.clip(img, 0, 1)

    # 绘图
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved RGB image to {save_path}")
        plt.close()
    else:
        plt.show()
        
def plot_numpy_rgb_image(img_np, title=None, save_path=None):
    """
    可视化一个 NumPy 格式的 RGB 图像。
    
    参数:
        img_np: NumPy 图像数组，形状应为 (H, W, 3)，通道顺序为 RGB
        title: 图像标题（可选）
        save_path: 若提供此路径，则保存图像而非显示（可选）
    """
    assert isinstance(img_np, np.ndarray), "输入必须是 numpy.ndarray"
    assert img_np.ndim == 3 and img_np.shape[2] == 3, f"图像必须为 (H, W, 3)，但得到的是 {img_np.shape}"
    
    # 若像素值为 uint8，则归一化到 [0, 1]
    if img_np.dtype == np.uint8:
        img_disp = img_np.astype(np.float32) / 255.0
    else:
        img_disp = np.clip(img_np, 0, 1)  # 避免 matplotlib 报错

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(img_disp)
    plt.axis('off')
    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图像已保存至 {save_path}")
        plt.close()
    else:
        plt.show()

def img_rel_preprocess(img, input_size=518):
    
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
    transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    return image

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = img / 255.0

    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406),
                                                            (0.229, 0.224, 0.225))
                                  ])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def remove_outliers_from_depth(depth_map, kernel_size=7, threshold=0.1):
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    valid_mask = (depth_map > 0)

    # 缩放到 [0, 255] 并转换为 uint8（也可选择 uint16）
    depth_max = depth_map.max()
    depth_uint8 = (depth_map / depth_max * 255).astype(np.uint8)

    # 中值滤波
    median_filtered = cv2.medianBlur(depth_uint8, kernel_size)

    # 还原回 float32
    median_filtered = median_filtered.astype(np.float32) / 255.0 * depth_max

    # 离群点剔除
    diff = np.abs(depth_map - median_filtered)
    outlier_mask = (diff > threshold) & valid_mask

    depth_cleaned = depth_map.copy()
    depth_cleaned[outlier_mask] = 0
    return depth_cleaned


def chromatic_transform(image):
    '''
        input:
            image: 
        
        return:
    '''
    # 
    d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    d_l = (np.random.rand(1) - 0.5) * 0.1 * 180
    d_s = (np.random.rand(1) - 0.5) * 0.1 * 180
    # convert the rgb to hls
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # add values to image, HLS
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(l + d_s, 0, 255)
    # convert the hls to rgb
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_image = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)
    return new_image


def add_noise(image, level=0.1):
    '''
    add noise to image
    
        Input:
            image
        Output:
            image
    '''
    # rand num
    r = np.random.rand(1)
    
    # gaussian noise
    if r < 0.9:
        row, col, ch = image.shape
        mean = 0 
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        #motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)
        
    return noisy.astype('uint8')
    

def process_depth(depth, camera_type=0, depth_min=0.3, depth_max=1.5, depth_norm=1.0):
    '''
        Input:
            dpeth: depth image
            camera_type: int in [0, 1, 2], optional, default: 0, the camera type;
                        - 0: no scale is applied;
                        - 1: scale 1000 (RealSense D415, RealSense D435, etc.);
                        - 2: scale 4000 (RealSense L515).
            depth_min, depth_max: int, optional, default: 0.3, 1.5, the min depth and the max depth;
            depth_norm: float, optional, default: 1.0, the depth normalization coefficient.
        Return:
            The depth image after scaling.
    '''
    scale_coeff = 1
    if camera_type == 1:
        scale_coeff = 1000
    elif camera_type == 2:
        scale_coeff = 4000

    depth = depth / scale_coeff
    depth[np.isnan(depth)] = 0.0
    depth = np.where(depth < depth_min, 0, depth)
    depth = np.where(depth > depth_max, 1.5, depth)
    depth = depth / depth_norm
    return depth


def handle_depth(depth, depth_gt, depth_gt_mask):
    
    depth[depth_gt_mask==1] = 0
    depth_gt_mask_uint8 = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
    depth_gt_mask_uint8[depth_gt_mask_uint8 != 0] = 1
    depth_uint8 = depth.copy() / depth.max() * 255
    depth_uint8 = np.array(depth_uint8, dtype=np.uint8)
    depth_uint8 = cv2.inpaint(depth_uint8, depth_gt_mask_uint8, 5, cv2.INPAINT_NS)
    depth_uint8 = np.array(depth_uint8, dtype=np.float32) / 255 * depth.max()
    
    mask_pixel_indices = np.array(np.where(depth_gt_mask == 1)).T
    dropout_size = int(mask_pixel_indices.shape[0] * 0.003) #这里的0.003是表明丢弃了0.3%的点云
    dropout_centers_indices = np.random.choice(mask_pixel_indices.shape[0], size=dropout_size)
    dropout_centers = mask_pixel_indices[dropout_centers_indices, :]
    x_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    y_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    angles = np.random.randint(0, 360, size=dropout_size)

    result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)

    for i in range(dropout_size // 2):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(x_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle,
                               startAngle=0, endAngle=360, color=1, thickness=-1)
        result_mask[tmp_mask == 1] = 1
        
    mask = np.logical_and(result_mask, depth_gt_mask_uint8)
    depth[mask == 1] = depth_uint8[mask == 1]

    result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
    
    for i in range(dropout_size - dropout_size // 2):
        center = dropout_centers[i + dropout_size // 2, :]
        x_radius = np.round(x_radii[i + dropout_size // 2]).astype(int)
        y_radius = np.round(y_radii[i + dropout_size // 2]).astype(int)
        angle = angles[i + dropout_size // 2]

        # get ellipse mask
        tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        result_mask[tmp_mask == 1] = 1

    mask = np.logical_and(result_mask, depth_gt_mask_uint8)
    depth[mask==1] = depth_gt[mask==1]
    
    return depth


def process_data(
    rgb, depth, depth_gt, depth_gt_mask, camera_intrinsics, scene_type='cluttered',
    camera_type=0, split='train', image_size=(720, 1280), depth_min=0.3, depth_max=1.5,
    depth_norm=10, use_aug=True, rgb_aug_prob=0.8, use_depth_aug=False, no_mask_depth=False,
    reldepth_model=None, **kwargs,
):
    """
    Process images and perform data augmentation.

    Parameters:
        rgb: array, required, the rgb image;
        depth: array, required, the original depth image;
        depth_gt: array, required, the ground-truth depth image;
        depth_gt_mask: array, required, the ground-truth depth image mask
        camera_intrinsics: array, required, the camera intrinsics of the image;
        scene_type: str in ['cluttered', 'isolated'], optional, default: 'cluttered', the scene type;
        camera_type: int in [0, 1, 2], optional, default: 0, the camera type;
            - 0: no scale is applied;
            - 1: scale 1000 (RealSense D415, RealSense D435, etc.);
            - 2: scale 4000 (RealSense L515).
        split: str in ['train', 'test'], optional, default: 'train', the split of the dataset;
        image_size: tuple of (int, int), optional, default: (720, 1280), the size of the image;
        depth_min, depth_max: float, optional, default: 0.1, 1.5, the min depth and the max depth
        depth_norm: float, optional, default: 1.0, the depth normalization coefficient;
        use_aug: bool, optional, default: True, whether use data augmentation;
        rgb_aug_prob: float, optional, default: 0.8, the rgb augmentation probability (only applies when use_aug is set to True).
    Returns:
        data_dict for training and testing.
    """
    
    depth_original = process_depth(depth.copy(), camera_type=camera_type, depth_min=depth_min,
                                  depth_max=depth_max, depth_norm=depth_norm)
    depth_gt_original = process_depth(depth_gt.copy(), camera_type=camera_type, depth_min=depth_min,
                                     depth_max=depth_max, depth_norm=depth_norm)
    depth_gt_mask_original = depth_gt_mask.copy()
    zero_mask_original = np.where(depth_gt_original < 0.01, False, True).astype(np.bool_)
    
    # depth processing
    depth = process_depth(depth, camera_type=camera_type, depth_min=depth_min, 
                          depth_max=depth_max, depth_norm=depth_norm)
    depth_gt = process_depth(depth_gt, camera_type=camera_type, depth_min=depth_min,
                             depth_max=depth_max, depth_norm=depth_norm)
    
    # cam_intrinsics = np.load('/home/cyf/YFTrans/datasets/transcg/transcg/camera_intrinsics/1-camIntrinsics-D435.npy')
    # pcd = depth_to_point_cloud(depth_gt, rgb, cam_intrinsics)
    # o3d.visualization.draw_geometries([pcd])
    rgb_relat = rgb.copy()
    
    # RGB augmentation.
    if split == 'train' and use_aug and np.random.rand(1) > 1 - rgb_aug_prob:
        rgb = chromatic_transform(rgb)
        rgb = add_noise(rgb)
    

    if split == 'train' and use_depth_aug:
        depth = handle_depth(depth.copy(), depth_gt.copy(), depth_gt_mask.copy())

    # Geometric augmentation
    if split == 'train' and use_aug:
        has_aug = False
        if np.random.rand(1) > 0.5:
            has_aug = True
            rgb = np.flip(rgb, axis = 0)
            depth = np.flip(depth, axis = 0)
            depth_gt = np.flip(depth_gt, axis = 0)
            depth_gt_mask = np.flip(depth_gt_mask, axis = 0)
        if np.random.rand(1) > 0.5:
            has_aug = True
            rgb = np.flip(rgb, axis = 1)
            depth = np.flip(depth, axis = 1)
            depth_gt = np.flip(depth_gt, axis = 1)
            depth_gt_mask = np.flip(depth_gt_mask, axis = 1)
        if has_aug:
            rgb = rgb.copy()
            depth = depth.copy()
            depth_gt = depth_gt.copy()
            depth_gt_mask = depth_gt_mask.copy()

    # RGB normalization
    rgb = rgb / 255.0
    rgb = rgb.transpose(2, 0, 1)

    # process scene mask
    scene_mask = (scene_type == 'cluttered')

    # zero mask
    neg_zero_mask = np.where(depth_gt < 0.01, 255, 0).astype(np.uint8)
    neg_zero_mask_dilated = cv2.dilate(neg_zero_mask, kernel = DILATION_KERNEL)
    neg_zero_mask[neg_zero_mask != 0] = 1
    neg_zero_mask_dilated[neg_zero_mask_dilated != 0] = 1
    zero_mask = np.logical_not(neg_zero_mask)
    zero_mask_dilated = np.logical_not(neg_zero_mask_dilated)

    # loss mask
    initial_loss_mask = np.logical_and(depth_gt_mask, zero_mask)
    initial_loss_mask_dilated = np.logical_and(depth_gt_mask, zero_mask_dilated)
    if scene_mask:
        loss_mask = initial_loss_mask
        loss_mask_dilated = initial_loss_mask_dilated
    else:
        loss_mask = zero_mask
        loss_mask_dilated = zero_mask_dilated
    
    # reldepth_model = kwargs.get('reldepth_model', None)
    if reldepth_model is not None:
        if reldepth_model == 'depthanything':
            # plot_numpy_rgb_image(rgb_relat)
            
            rgb_relat = cv2.resize(rgb_relat, (518, 686))
            rgb_relat = img_rel_preprocess(rgb_relat)
            # print(rgb_relat.shape)
            # plot_rgb_image(rgb_relat.transpose(1, 2, 0))
            
        elif reldepth_model == 'leres':
            # print(rgb_relat.shape)
            rgb_c = rgb_relat[:, :, ::-1].copy()
            # plot_numpy_rgb_image(rgb_relat)
            resized = cv2.resize(rgb_c, (448, 448))
            # print(rgb_relat.shape)
            # plot_rgb_image(rgb_relat)
            # rgb_relat = img_rel_preprocess(rgb_relat, 448)
            # print(rgb_relat.shape)
            rgb_relat = scale_torch(resized)#.permute(0, 2, 1)
            # print(rgb_relat.shape)
            # print(rgb_relat)
            # plot_rgb_image(rgb_relat)
    
    
    if no_mask_depth is True:
        depth = depth * (1 - depth_gt_mask) #no trans depth
        print('no mask depth')
    
    # cam_intrinsics = np.load('/home/cyf/YFTrans/datasets/transcg/transcg/camera_intrinsics/1-camIntrinsics-D435.npy')
    # pcd = depth_to_point_cloud_no_color(depth, cam_intrinsics)
    # o3d.visualization.draw_geometries([pcd])
    
    # plot_image_process(depth, depth_gt, depth, rgb, depth_gt_mask, initial_loss_mask)
        
    data_dict = {
        'rgb': torch.FloatTensor(rgb),
        'rgb_relat': torch.FloatTensor(rgb_relat),
        'depth': torch.FloatTensor(depth),
        'depth_min': torch.tensor(depth_min),
        'depth_max': torch.tensor(depth_max),  #TDC少了max和min
        'depth_gt': torch.FloatTensor(depth_gt),
        'depth_gt_mask': torch.BoolTensor(depth_gt_mask),
        'scene_mask': torch.tensor(scene_mask),
        'zero_mask': torch.BoolTensor(zero_mask),
        'zero_mask_dilated': torch.BoolTensor(zero_mask_dilated),
        'initial_loss_mask': torch.BoolTensor(initial_loss_mask),
        'initial_loss_mask_dilated': torch.BoolTensor(initial_loss_mask_dilated),
        'loss_mask': torch.BoolTensor(loss_mask),
        'loss_mask_dilated': torch.BoolTensor(loss_mask_dilated),
        'depth_original': torch.FloatTensor(depth_original),
        'depth_gt_original': torch.FloatTensor(depth_gt_original),
        'depth_gt_mask_original': torch.BoolTensor(depth_gt_mask_original),
        'zero_mask_original': torch.BoolTensor(zero_mask_original),
        'fx': torch.tensor(camera_intrinsics[0, 0]),
        'fy': torch.tensor(camera_intrinsics[1, 1]),
        'cx': torch.tensor(camera_intrinsics[0, 2]),
        'cy': torch.tensor(camera_intrinsics[1, 2])
    }
    
    data_dict['depth_gt_sn'] = get_surface_normal_from_depth(
        data_dict['depth_gt'].unsqueeze(0), data_dict['fx'].unsqueeze(0), data_dict['fy'].unsqueeze(0),
        data_dict['cx'].unsqueeze(0), data_dict['cy'].unsqueeze(0)
    ).squeeze(0)
    
    return data_dict


def exr_loader(exr_path, ndim=3, ndim_representation=['R', 'G', 'B']):
    """
    Loads a .exr file as a numpy array.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py.

    Parameters
    ----------

    exr_path: path to the exr file
    
    ndim: number of channels that should be in returned array. Valid values are 1 and 3.
        - if ndim=1, only the 'R' channel is taken from exr file;
        - if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file. The exr file must have 3 channels in this case.
    
    depth_representation: list of str, the representation of channels, default = ['R', 'G', 'B'].
    
    Returns
    -------

    numpy.ndarray (dtype=np.float32).
        - If ndim=1, shape is (height x width);
        - If ndim=3, shape is (3 x height x width)
    """
    exr_file = OpenEXR.InputFile(exr_path)
    cm_dw = exr_file.header()['dataWindow']
    size = [cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    assert ndim == len(ndim_representation), "ndim should match ndim_representation"

    if ndim == 3:
        #read channdels individualy
        allchannels = []
        for c in ndim_representation:
            #transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)
    
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel(ndim_representation[0], pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr