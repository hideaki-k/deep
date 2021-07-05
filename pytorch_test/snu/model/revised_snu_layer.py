# -*- coding: utf-8 -*-

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.parameter as Parameter

import torch.optim as optim
import torchvision
import numpy as np
from torch import cuda

import numpy 
from . import step_func
#import step_func
# Gated_SNU ==　畳み込みなし
class Gated_SNU(nn.Module):
    def __init__(self, k, rec=True, l_tau=0.8, initial_bias=-0.5, gpu=True):
        super(Gated_SNU,self).__init__()

        self.rec = rec
        self.gpu = gpu
        self.s = None
        self.y = None
        self.l_tau = l_tau # 膜電位忘却定数
        self.initial_bias = initial_bias 
        self.k = k
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
        self.Wx = nn.Linear(k,k)
        #print("self.rec",self.rec)
        if self.rec==True:
            self.Wi = nn.Linear(k,k)
            self.Wf = nn.Linear(k,k)
            self.Wy = nn.Linear(k,k)
            self.Ri = nn.Linear(k,k)
            self.Rf = nn.Linear(k,k)
            
    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y       
    
    def initialize_state(self, shape):
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.s = torch.zeros((shape[0], self.k ),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.k ),device=device,dtype=dtype)
    
    def forward(self, x):
        if self.s is None:
            self.initialize_state(x.shape)
        
        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()

        if self.rec==True:
            print("####")
            # 入力ゲート
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
            # 忘却ゲート
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            # 膜電位
            s = F.elu(self.Wx(x) + i * self.Wy(self.y) + f * self.s * (1-self.y))
            
        # 膜電位
        s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))



        axis = 0
        bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
        bias = s + self.b
        y = step_func.spike_fn(bias)
        self.s = s
        self.y = y

        return y


class Gated_Conv_SNU(nn.Module):
    def __init__(self,channelIn=1, channelOut=1, kernel_size=5, padding=2, initial_bias=-0.5, gpu=True):
        super(Gated_Conv_SNU,self).__init__()

        self.channelIn=channelIn
        self.channelOut=channelOut
        self.kernel_size=kernel_size
        self.padding=padding
        self.gpu = gpu
        self.s = None
        self.y = None

        self.initial_bias = initial_bias 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))

        self.Wx = nn.Conv2d(self.channelIn,self.channelOut,self.kernel_size,stride=1,padding=self.padding)
        self.Wi = nn.Conv2d(self.channelIn,self.channelOut,self.kernel_size,stride=1,padding=self.padding)
        self.Wf = nn.Conv2d(self.channelIn,self.channelOut,self.kernel_size,stride=1,padding=self.padding)
        self.Wy = nn.Conv2d(self.channelIn,self.channelOut,self.kernel_size,stride=1,padding=self.padding)
        self.Ri = nn.Conv2d(self.channelIn,self.channelOut,self.kernel_size,stride=1,padding=self.padding)
        self.Rf = nn.Conv2d(self.channelIn,self.channelOut,self.kernel_size,stride=1,padding=self.padding)
        torch.nn.init.xavier_uniform_(self.Wx.weight)
        torch.nn.init.xavier_uniform_(self.Wi.weight)
        torch.nn.init.xavier_uniform_(self.Wf.weight)
        torch.nn.init.xavier_uniform_(self.Wy.weight)
        torch.nn.init.xavier_uniform_(self.Ri.weight)
        torch.nn.init.xavier_uniform_(self.Wf.weight)

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y       
    
    def initialize_state(self, shape):
        dtype = torch.float
        hei = shape[2]
        wid = shape[3]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.s = torch.zeros((shape[0], self.channelOut,hei,wid ),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.channelOut,hei,wid ),device=device,dtype=dtype)
    
    def forward(self, x):
        if self.s is None:
            self.initialize_state(x.shape)
        
        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()


        i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
        # 忘却ゲート
        f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
        # 膜電位
        s = F.elu(self.Wx(x) + i * self.Wy(self.y) + f * self.s * (1-self.y))




        axis = 0
        bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
        bias = s + self.b
        y = step_func.spike_fn(bias)
        self.s = s
        self.y = y

        return y
