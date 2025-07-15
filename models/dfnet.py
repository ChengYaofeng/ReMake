import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import xavier_init
from einops import rearrange


class DenseBlock(nn.Module):
    def __init__(self, in_channels, layer_num, k, with_bn=False):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.layer_num = layer_num
        self.with_bn = with_bn
        self.layers = []
        for i in range(layer_num):
            layer_in_channels = in_channels + i * k + 1
            single_layer = []
            # conv1
            conv1 = nn.Conv2d(layer_in_channels, k*4, kernel_size=1, stride=1)
            xavier_init(conv1)
            single_layer.append(conv1)
            if with_bn:
                single_layer.append(nn.BatchNorm2d(k * 4))
            single_layer.append(nn.ReLU(True))
            # conv2
            conv2 = nn.Conv2d(k * 4, k, kernel_size = 3, stride = 1, padding = 1)
            xavier_init(conv2)
            single_layer.append(conv2)
            if with_bn:
                single_layer.append(nn.BatchNorm2d(k))
            single_layer.append(nn.ReLU(True))
            self.layers.append(nn.Sequential(*single_layer))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        h = x
        hs = [h]
        for i in range(self.layer_num):
            if i != 0:
                h = torch.cat(hs, dim=1)
            h = self.layers[i](h)
            if i != self.layer_num - 1:
                hs.append(h)
        return h
    

class DenseUpsamplingConvolution(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DenseUpsamplingConvolution, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inplanes, planes*upscale_factor*upscale_factor, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes*upscale_factor*upscale_factor),
            nn.ReLU(True)
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, x):
        x = self.layer(x)
        x = self.pixel_shuffle(x)
        return x
            

class DFNet(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, L=5, k=12, use_DUC=True, **kwargs):
        super(DFNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.layer_num = L
        self.k = k
        self.use_DUC = use_DUC
        #First
        self.first_layer = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense1: skip
        self.dense1s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.dense1s = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn=True)
        self.dense1s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
        )
        # Dense1: Normal
        self.dense1_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
        )
        self.dense1 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn=True)
        self.dense1_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # Dense2: skip
        self.dense2s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2s = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.dense2s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense2: normal
        self.dense2_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense2 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.dense2_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: skip
        self.dense3s_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3s = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn=True)
        self.dense3s_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense3: normal
        self.dense3_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense3 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.dense3_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense4
        self.dense4_conv1 = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.dense4 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.dense4_conv2 = nn.Sequential(
            nn.Conv2d(self.k, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        # DUC upsample 1
        self.updense1_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense1 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.updense1_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 2
        self.updense2_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense2 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.updense2_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 3
        self.updense3_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense3 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn = True)
        self.updense3_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # DUC upsample 4
        self.updense4_conv = nn.Sequential(
            nn.Conv2d(self.hidden_channels * 2, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True)
        )
        self.updense4 = DenseBlock(self.hidden_channels, self.layer_num, self.k, with_bn=True)
        self.updense4_duc = self._make_upconv(self.k, self.hidden_channels, upscale_factor = 2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size = 1, stride = 1)
        )
    
    def _make_upconv(self, in_channels, out_channels, upscale_factor=2):
        if self.use_DUC:
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor = upscale_factor)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upscale_factor, stride=upscale_factor, padding=0, output_padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
            
    
    def forward(self, rgb, depth):
        # 720 x 1280 (rgb, depth) -> (360, 640)
        n, h, w = depth.shape
        depth = depth.view(n, 1, h, w)
        h = self.first_layer(torch.cat((rgb, depth), dim=1))
        
        # dense1: 360 x 640 (h, depth1) -> 180 x 320 (h, depth2)
        depth1 = F.interpolate(depth, scale_factor=0.5, mode='nearest')
        # dense1: skip
        h_d1s = self.dense1s_conv1(h)
        h_d1s = self.dense1s(torch.cat((h_d1s, depth1), dim = 1))
        h_d1s = self.dense1s_conv2(h_d1s)
        # dense1: normal
        h = self.dense1_conv1(h)
        h = self.dense1(torch.cat((h, depth1), dim = 1))
        h = self.dense1_conv2(h)
        
        # dense2: 180 x 320 (h, depth2) -> 90 x 160 (h, depth3)
        depth2 = F.interpolate(depth1, scale_factor = 0.5, mode = "nearest")
        # dense2: skip
        h_d2s = self.dense2s_conv1(h)
        h_d2s = self.dense2s(torch.cat((h_d2s, depth2), dim = 1))
        h_d2s = self.dense2s_conv2(h_d2s)
        # dense2: normal
        h = self.dense2_conv1(h)
        h = self.dense2(torch.cat((h, depth2), dim = 1))
        h = self.dense2_conv2(h)
        
        # dense3: 90 x 160 (h, depth3) -> 45 x 80 (h, depth4)
        depth3 = F.interpolate(depth2, scale_factor = 0.5, mode = "nearest")
        # dense3: skip
        h_d3s = self.dense3s_conv1(h)
        h_d3s = self.dense3s(torch.cat((h_d3s, depth3), dim = 1))
        h_d3s = self.dense3s_conv2(h_d3s)
        # dense3: normal
        h = self.dense3_conv1(h)
        h = self.dense3(torch.cat((h, depth3), dim = 1))
        h = self.dense3_conv2(h)

        # dense4: 45 x 80
        depth4 = F.interpolate(depth3, scale_factor = 0.5, mode = "nearest")
        h = self.dense4_conv1(h)
        h = self.dense4(torch.cat((h, depth4), dim = 1))
        h = self.dense4_conv2(h)

        # updense1: 45 x 80 -> 90 x 160
        h = self.updense1_conv(h)
        h = self.updense1(torch.cat((h, depth4), dim = 1))
        h = self.updense1_duc(h)

        # updense2: 90 x 160 -> 180 x 320
        h = torch.cat((h, h_d3s), dim = 1)
        h = self.updense2_conv(h)
        h = self.updense2(torch.cat((h, depth3), dim = 1))
        h = self.updense2_duc(h)

        # updense3: 180 x 320 -> 360 x 640
        h = torch.cat((h, h_d2s), dim = 1)
        h = self.updense3_conv(h)
        h = self.updense3(torch.cat((h, depth2), dim = 1))
        h = self.updense3_duc(h)

        # updense4: 360 x 640 -> 720 x 1280
        h = torch.cat((h, h_d1s), dim = 1)
        h = self.updense4_conv(h)
        h = self.updense4(torch.cat((h, depth1), dim = 1))
        h = self.updense4_duc(h)

        # final
        h = self.final(h)

        return rearrange(h, 'n 1 h w -> n h w')
    
    @classmethod
    def build(cls, **kwargs):
 
        print('Building Encoder-Decoder model..', end='')
        m = cls(**kwargs)
        print('Done.')
        return m