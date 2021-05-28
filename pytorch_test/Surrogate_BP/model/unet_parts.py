import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.append('C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu\model')
import snu_layer

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv1 = snu_layer.Conv_SNU(in_channels, mid_channels, kernel_size=3, padding=1),
        self.double_conv1.Conv_SNU(mid_channels,out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = self.double_conv1(x)
        x2 = self.double_conv2(x1)
        return x2
         
    def reset_state(self):
        self.double_conv1.reset_state()
        self.double_conv2.reset_state()
        

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.l1 = nn.MaxPool2d(2),
        self.l2 = DoubleConv(in_channels, out_channels)
        

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        return x2
    
    def reset_state(self):
        self.l2.reset_state()

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = snu_layer.tConv_SNU(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY //2, diffY - diffY //2])
        
        x = torch.cat([x2, x1],dim=1)
        return self.conv(x)

    def reset_state(self):
        self.up.reset_state()
        self.conv.reset_state()
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = snu_layer.Conv_SNU(in_channels, out_channels, kernel_size=1)

    def forward(self,x):
        return self.conv(x)

    def reset_state(self):
        self.conv.reset_state()