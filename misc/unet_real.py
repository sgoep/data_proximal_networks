# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.custom_layers_real import proximal_layer, null_space_layer, range_layer
from config import config
from misc.radon_operator import RadonOperator
import numpy as np
import scipy.io

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    # def __init__(self, n_channels, n_classes, T, bilinear=True):
    def __init__(self, n_channels, n_classes, CONSTRAINT, null_space_network, bilinear=True):
    # def __init__(self, n_channels, n_classes, TYPE, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.tanh = nn.Tanh()
        self.CONSTRAINT = CONSTRAINT


        sample = "01a"
        ct_data = scipy.io.loadmat(f'data_htc2022/htc2022_test_data/htc2022_{sample}_limited.mat')
        ct_data = ct_data["CtDataLimited"][0][0]
        angles = ct_data["parameters"]["angles"][0, 0][0]

        Aop = RadonOperator(angles)
        self.A = lambda x: Aop.forward(x)
        # AT = lambda y: Aop.backward(y)
        self.B = lambda z: Aop.fbp(z)
        
        self.proximal_layer   = proximal_layer()
        self.null_space_layer = null_space_layer()
        self.range_layer      = range_layer()
        
        self.null_space_network = null_space_network
  
        
                    
    def forward(self, X):
        x0  = X
        
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        if not self.null_space_network:
            x_res = x
        else:
            if self.CONSTRAINT:
                x_res = self.proximal_layer(x, self.A, self.B)
                x_res += self.null_space_layer(x, self.A, self.B)
            else:
                x_res = self.range_layer(x, self.A, self.B)
                x_res += self.null_space_layer(x, self.A, self.B)
        
        return x0 + x_res, x_res

# %%

# x = np.linspace(-1, 1, 1000)
# alpha = 0.5
# phi = lambda x: x/np.sqrt(1+alpha*np.linalg.norm(x)**2)
# plt.plot(phi(x))
        
     