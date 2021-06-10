import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *
import matplotlib.pyplot as plt
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, num_time=10, gpu=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.num_time = num_time

        self.inc = DoubleConv_1(n_channels, 64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)
        self.up1 = Up(1024, 512//factor,bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor,bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def _reset_state(self):
        self.inc.reset_state()
        self.down1.reset_state()
        self.down2.reset_state()
        self.down3.reset_state()
        self.down4.reset_state()
        self.up1.reset_state()
        self.up2.reset_state()
        self.up3.reset_state()
        self.up4.reset_state()
        self.outc.reset_state()

    def forward(self, x_timeline):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        out = torch.zeros((x_timeline.shape[0],1,64,64), device=device, dtype=torch.float)
        out_rec = [out]

        self._reset_state()

        for t in range(self.num_time):

            #print('x shape:',x_timeline.shape) #([2, 512, 512, 21])
            x_t = x_timeline[:,:,:,t]
            x_t_=x_t.to('cpu').detach().numpy().copy()
            print("x_t_ shape",x_t_.shape)
            plt.imshow(x_t_[0,:,:])
            plt.show()
            #print("x_t",x_t.shape)
            x_t = x_t.reshape((len(x_t),1,64,64)) 
            #print('x_t shape',x_t.shape)
            x1 = self.inc(x_t) #([2, 64, 128, 128])

            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            out = self.outc(x)
            #print("====")
            #print("out",out.shape)
            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        m = torch.sum(out_rec,1)

        return m




