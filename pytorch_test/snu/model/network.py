# -*- coding: utf-8 -*-
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from . import snu_layer

# Network definition
class SNU_Network(torch.nn.Module):
    def __init__(self, n_in=784, n_mid=256, n_out=10,
                 num_time=20, l_tau=0.8, soft=False, gpu=False,
                 test_mode=False):
        super(SNU_Network, self).__init__()
        
        self.l1 = snu_layer.SNU(n_in, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l2 = snu_layer.SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l3 = snu_layer.SNU(n_mid, n_mid, l_tau=l_tau, soft=soft, gpu=gpu)
        self.l4 = snu_layer.SNU(n_mid, n_out, l_tau=l_tau, soft=soft, gpu=gpu)
        
        self.n_out = n_out
        self.num_time = num_time
        self.gamma = (1/(num_time*n_out))*1e-3
        self.test_mode = test_mode

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        
    def forward(self, x, y):
        loss = None
        accuracy = None
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out = torch.zeros((256,self.n_out), device=device, dtype=dtype)
        out_rec = [out]
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()
        self._reset_state()
        
        if self.test_mode == True:
            h1_list = []
            h2_list = []
            h3_list = []
            out_list = []
        
        for t in range(self.num_time):
            print("**********************")
            print("t :",t)
            x_t = x[:,t]  #torch.Size([256, 784])
            print("x_t[0].shape",x_t[0].shape)
            print("x_t[0]",x_t[0])
            h1 = self.l1(x_t) # torch.Size([256, 256])

            h2 = self.l2(h1) #h2.shape: torch.Size([256, 256])

            h3 = self.l3(h2)

            out = self.l4(h3) #out.shape torch.Size([256, 10]) # [バッチサイズ,output.shape]
            print("out[0].shape",out[0].shape) #out[0].shape torch.Size([10])
            print("out[0]:",out[0])  #tensor([1., 0., 1., 0., 1., 0., 1., 1., 0., 1.], device='cuda:0',

            
            if self.test_mode == True:
                h1_list.append(h1)
                h2_list.append(h2)
                h3_list.append(h3)
                out_list.append(out)
            
            #sum_out = out if sum_out is None else sum_out + out
            out_rec.append(out)
        print("///////////////////////")
        out_rec = torch.stack(out_rec,dim=1)
        print("out_rec.shape",out_rec.shape) #out_rec.shape torch.Size([256, 11, 10]) ([バッチ,時間,分類])
        #m,_=torch.sum(out_rec,1)
        m =torch.sum(out_rec,1) #m.shape: torch.Size([256, 10])
        print("m.shape:",m.shape)
        print("m",m)
        
        print("out_rec.shape",out_rec.shape)
        print("y.shape",y.shape)
        

        print("///////////////////////")
        criterion = nn.CrossEntropyLoss()
        loss = criterion(m, y)
        #print("loss",loss)
        #loss += self.gamma*torch.sum(sum_out**2)
        #print("gamma loss",loss)
        
        

        
        if self.test_mode == True:
            return loss, accuracy, h1_list, h2_list, h3_list, out_list
        else:
            return loss, m