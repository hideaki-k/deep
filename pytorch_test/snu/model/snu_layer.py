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

class SNU(nn.Module):
    def __init__(self, n_in, n_out, l_tau=0.8, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True):
        super(SNU,self).__init__()
        
        self.n_out = n_out
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        
        #self.w1 = torch.empty((n_in, n_out),  device=device, dtype=dtype, requires_grad=True)
        #torch.nn.init.normal_(self.w1, mean=0.0)
        
        #self.Wx = torch.einsum("abc,cd->abd", (x_data, w1))
        self.Wx = nn.Linear(4374, n_out, bias=False).to(device)
        #nn.init.uniform_(self.Wx.weight, -0.1, 0.1) #3.0
        torch.nn.init.xavier_uniform_(self.Wx.weight)

    

        if nobias:
            self.b = None
        else:

            #print("initial_bias",initial_bias)
            device = torch.device(device)
            
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
            #print("self.b",self.b)
                            
    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.n_out),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.n_out),device=device,dtype=dtype)
              
    
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)


        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print("x in snu.shape",x.shape) #x in snu.shape torch.Size([256, 784])        
        #print("self.Wx(x).shape",self.Wx(x).shape)
        #print("self.s.shape : ",self.s.shape)
        s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        #print("s : ",s)

        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = F.sigmoid(bias_)

        else:
            axis = 0

            #print("s.shape:", s.shape)
            #print("self.b.shape:", self.b.shape)
            #print("self.initial_bias.shape:",self.initial_bias.shape)
            #print("self.b.shape !!!!!!!!!!!!!!!! ", self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)].shape)
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            #print("bias:",bias)
            #print("s in snu:",s)
            bias = s + self.b
            #bias = s + self.initial_bias
            
            #print("bias in snu:",bias)
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y

class Conv_SNU(nn.Module):
    def __init__(self, n_in, n_out, filter, l_tau=0.8, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True):
        super(Conv_SNU,self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.filter = filter
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        

        self.Wx = nn.Conv2d(n_in, n_out, filter, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
        torch.nn.init.xavier_uniform_(self.Wx.weight)

        if nobias:
            self.b = None
        else:

            #print("initial_bias",initial_bias)
            device = torch.device(device)
            
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
            #print("self.b",self.b)

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.n_out, 55, 55),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.n_out, 55, 55),device=device,dtype=dtype)
        
    
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)


        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print("x in snu:",x) 
        print("x in snu.shape",x.shape) #x in snu.shape torch.Size([256, 784])
        print("self.Wx(x).shape",self.Wx(x).shape)
        
        print("self.Wx(x).shape",self.Wx(x).shape)
        print("self.s.shape : ",self.s.shape)
        #print( " self.l_tau * self.s * (1-self.y)):",self.l_tau * self.s * (1-self.y))
        #print("self.Wx(x):",self.Wx(x))
        
        s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        

        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = F.sigmoid(bias_)

        else:
            axis = 0

            #print("s.shape:", s.shape)
            #print("self.b.shape:", self.b.shape)
            #print("self.initial_bias.shape:",self.initial_bias.shape)
            #print("self.b.shape !!!!!!!!!!!!!!!! ", self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)].shape)
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            #print("bias:",bias)
            #print("s in snu:",s)
            bias = s + self.b
            #bias = s + self.initial_bias
            
            #print("bias in snu:",bias)
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y


class tConv_SNU(nn.Module):
    def __init__(self, n_in, n_out, filter, l_tau=0.8, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True):
        super(Conv_SNU,self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.filter = filter
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        

        self.Wx = nn.ConvTranspose2d(n_in, n_out, filter, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
        torch.nn.init.xavier_uniform_(self.Wx.weight)

        if nobias:
            self.b = None
        else:

            #print("initial_bias",initial_bias)
            device = torch.device(device)
            
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
            #print("self.b",self.b)

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.n_out, 55, 55),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.n_out, 55, 55),device=device,dtype=dtype)
        
    
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)


        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print("x in snu:",x) 
        print("x in snu.shape",x.shape) #x in snu.shape torch.Size([256, 784])
        print("self.Wx(x).shape",self.Wx(x).shape)
        
        print("self.Wx(x).shape",self.Wx(x).shape)
        print("self.s.shape : ",self.s.shape)
        #print( " self.l_tau * self.s * (1-self.y)):",self.l_tau * self.s * (1-self.y))
        #print("self.Wx(x):",self.Wx(x))
        
        s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        

        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = F.sigmoid(bias_)

        else:
            axis = 0

            #print("s.shape:", s.shape)
            #print("self.b.shape:", self.b.shape)
            #print("self.initial_bias.shape:",self.initial_bias.shape)
            #print("self.b.shape !!!!!!!!!!!!!!!! ", self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)].shape)
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            #print("bias:",bias)
            #print("s in snu:",s)
            bias = s + self.b
            #bias = s + self.initial_bias
            
            #print("bias in snu:",bias)
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y