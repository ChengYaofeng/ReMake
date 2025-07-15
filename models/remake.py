
import torch
import torch.nn as nn
# from .resnet import resnet18
import torch.nn.functional as F
from models.swin_transformer import SwinTransformer
import math
from time import perf_counter
from utils.visualization import plot_realat_depth, run_gradcam_on_encoder_img
from models.resnet import resnet18
import matplotlib.pyplot as plt
import torchvision

def visualize_feature_map(feat, title="Feature Map", nrow=8):
    """可视化一个单层特征图的通道"""
    feat = feat[0, :8].unsqueeze(1)  # 取 batch 中第一个样本，变成 [C, 1, H, W]
    feat_grid = torchvision.utils.make_grid(feat, nrow=nrow, normalize=True, scale_each=True)

    img = feat_grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


class UpSample(nn.Module):
    def __init__(self, input_features, output_features):
        super(UpSample, self).__init__()
        
        self._net = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_features),
            nn.ReLU(True),
            nn.Conv2d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_features),
            nn.ReLU(True)
        )
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(input_features, output_features, kernel_size = 2, stride = 2, padding = 0, output_padding = 0),
            nn.BatchNorm2d(output_features, output_features),
            nn.ReLU(True)
        )
        
    def forward(self, x, y):
        if y == None:
            conv_x = self._net(x) + x
        else:
            conv_x = self._net(torch.cat([x, y], dim=1)) + torch.cat([x, y], dim=1)
            
        return self.net(conv_x)

class Decoder(nn.Module):
    def __init__(self, num_features=128, lambda_val=1, res=True):
        super(Decoder, self).__init__()
        
        self.up_sample_layer1 = UpSample(192, num_features)
        self.up_sample_layer2 = UpSample(96 + num_features, num_features)
        self.up_sample_layer3 = UpSample(48 + num_features, num_features)
        self.up_sample_layer4 = UpSample(24 + num_features, num_features // 2)


    
    def forward(self, features):
        x1, x2, x3, x4 = features[3], features[2], features[1], features[0]
        # 192, 96, 48, 24
        f1 = self.up_sample_layer1(x1, None)
        # visualize_feature_map(f1, "Decoder Layer 1 Output")
        f2 = self.up_sample_layer2(x2, f1)
        # visualize_feature_map(f2, "Decoder Layer 2 Output")
        f3 = self.up_sample_layer3(x3, f2)
        # visualize_feature_map(f3, "Decoder Layer 3 Output")
        f4 = self.up_sample_layer4(x4, f3)
        # visualize_feature_map(f4, "Decoder Layer 4 Output")
        
        return f4
        


class ReMak(nn.Module):
    def __init__(self, lambda_val = 1, res = True):
        super(ReMak, self).__init__()
        
        self.encoder_img = SwinTransformer(patch_size=2, in_chans=4, embed_dim=24)
        self.encoder_rel = SwinTransformer(patch_size=2, in_chans=1, embed_dim=24)
        self.resnet = resnet18(pretrained=False)
        
        self.decoder = Decoder(num_features=128, lambda_val=lambda_val, res=res)
        
        self.final = self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(True)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, img, relative_depth, depth, mask=None, **kwargs):
        
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)
        
        # plot_realat_depth(relative_depth[0].squeeze())
        relative_depth = relative_depth.view(n, 1, h, w)
        mask = mask.view(n, 1, h, w)
        
        encoder_depth = self.resnet(depth)
        encoder_rgb = self.encoder_img(torch.cat((img, mask), dim=1))
        encoder_rel = self.encoder_rel(relative_depth)

        # for i in range(1):
        #     feat = encoder_rgb[i][0, :8].unsqueeze(1)
        #     feat_grid = torchvision.utils.make_grid(feat, nrow=4, normalize=True, scale_each=True)

        #     if feat_grid.ndim == 3:
        #         img_to_show = feat_grid.permute(1, 2, 0).cpu().numpy()
        #     else:
        #         img_to_show = feat_grid.cpu().numpy()

        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(img_to_show)
        #     plt.title("Encoder Feature Map Grid")
        #     plt.axis("off")
        #     plt.show()

        # for i in range(1):
        #     feat = encoder_rel[i][0, :8].unsqueeze(1)
        #     feat_grid = torchvision.utils.make_grid(feat, nrow=4, normalize=True, scale_each=True)

        #     if feat_grid.ndim == 3:
        #         img_to_show = feat_grid.permute(1, 2, 0).cpu().numpy()
        #     else:
        #         img_to_show = feat_grid.cpu().numpy()

        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(img_to_show)
        #     plt.title("Encoder Feature Map Grid")
        #     plt.axis("off")
        #     plt.show()
        
        # for i in range(1):
        #     feat = encoder_depth[i][0, :8].unsqueeze(1)
        #     feat_grid = torchvision.utils.make_grid(feat, nrow=4, normalize=True, scale_each=True)

        #     if feat_grid.ndim == 3:
        #         img_to_show = feat_grid.permute(1, 2, 0).cpu().numpy()
        #     else:
        #         img_to_show = feat_grid.cpu().numpy()

        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(img_to_show)
        #     plt.title("Encoder Feature Map Grid Depth")
        #     plt.axis("off")
        #     plt.show()

        encoder_rgb[0] = encoder_rgb[0] + encoder_depth[0] + encoder_rel[0]
        encoder_rgb[1] = encoder_rgb[1] + encoder_depth[1] + encoder_rel[1]
        encoder_rgb[2] = encoder_rgb[2] + encoder_depth[2] + encoder_rel[2]
        encoder_rgb[3] = encoder_rgb[3] + encoder_depth[3] + encoder_rel[3]

        decoder_x = self.decoder(encoder_rgb)
        out = self.final(decoder_x)
        
        return out
    
    
    @classmethod
    def build(cls, **kwargs):
 
        print('Building Encoder-Decoder model..', end='')
        m = cls(**kwargs)
        print('Done.')
        return m