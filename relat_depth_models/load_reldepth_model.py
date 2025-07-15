import torch
import os
from relat_depth_models import DepthAnythingV2, RelDepthModel, strip_prefix_if_present

def load_depthanything(device):

    # INPUT_SIZE = 518  # Default input size

    # Choose encoder type
    ENCODER_TYPE = 'vits'  # Choose from ['vits', 'vitb', 'vitl', 'vitg']

    # Define model configs
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize model
    depth_anything = DepthAnythingV2(**model_configs[ENCODER_TYPE])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{ENCODER_TYPE}.pth', map_location='cuda'))
    for param in depth_anything.parameters():
        param.requires_grad = False
    depth_anything = depth_anything.to(device).eval()
    return depth_anything

def load_leres(device):
    reldepth_model = RelDepthModel(backbone='resnext101') #resnet50  resnext101
    
    for param in reldepth_model.parameters():
        param.requires_grad = False
    reldepth_model = reldepth_model.to(device).eval()

    # load checkpoint res101.pth
    if os.path.isfile('res101.pth'):
        print("loading checkpoint %s" % 'res101.pth') #res50
        checkpoint = torch.load('res101.pth')

        reldepth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                                    strict=True)
    return reldepth_model

def load_reldepth_model(model_name, device):
    
    if model_name == 'depthanything':
        reldepth_model = load_depthanything(device)
    elif model_name == 'leres':
        reldepth_model = load_leres(device)
    else:
        raise NameError('no such rel depth model')
    
    return reldepth_model