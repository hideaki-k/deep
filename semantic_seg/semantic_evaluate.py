import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import os 
from tqdm import tqdm
from rectangle_builder import rectangle,test_img
import sys
sys.path.append(r"C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu")
from model import snu_layer
from model import network
from tqdm import tqdm
from mp4_rec import record, rectangle_record
import pandas as pd
import scipy.io
from torchsummary import summary

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
       
        self.df = pd.read_csv(csv_file)
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        file = self.df['id'][i]
        image = scipy.io.loadmat(file)
        label = scipy.io.loadmat(self.df['label'][i])

        image = image['name']
        label = label['name']
        #print("image : ",image.shape)
        #print("label : ",label.shape)
        image = image.reshape(4096,20)
        #print("image : ",image.shape)
        image = image.astype(np.float32)
        #label = label.astype(np.int64)
        #label = torch.tensor(label,dtype =torch.int64 )
        label = label.reshape(4096)
        label = label.astype(np.float32)
        return image, label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     
model = network.SNU_Network(gpu=True)
model = model.to(device)
model.load_state_dict(torch.load("models/models_state_dict.pth"))
print("load model")

 # test_img : 評価用画像生成
inputs_, label_ = test_img() 
inputs_ = inputs_.to(device)
label_ = label_.to(device)
print("inputs_",inputs_.shape) # 
print("label_",label_.shape)

label, pred, result = model(inputs_, label_)
print("result shape : ",result.shape) #torch.Size([128, 21, 1, 64, 64])
print("inputs_ shape : ",inputs_.shape) #torch.Size([128, 4096, 20])

device2 = torch.device('cpu')
result = result.to(device2)
result = result.detach().clone().numpy()
inputs_ = inputs_.to(device2)
inputs_ = inputs_.detach().clone().numpy() 
# MP4  レコード

data_id = 2
num_time = 20
rectangle_record(inputs_,num_time=num_time, data_id=data_id)
record(result,num_time=num_time,data_id=data_id)