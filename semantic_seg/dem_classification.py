import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from rectangle_builder import rectangle,test_img
import sys
sys.path.append(r"C:\Users\aki\Documents\GitHub\deep\pytorch_test\snu")
from model import snu_layer
from model import network
from tqdm import tqdm
from mp4_rec import record, rectangle_record
from PIL import Image
import scipy.io

class TrainLoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
       
        self.df = pd.read_csv(csv_file)
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        file = self.df['id'][i]
        label = np.array(self.df['label'][i])
        image = scipy.io.loadmat(file)
        #print("label : ",label)
        print("image : ",image['time_data'])
        #label = torch.tensor(label, dtype=torch.float32)
        image = torch.tensor(image['time_data'])
 
        return image, label
     
 


train_dataset = TrainLoadDataset(csv_file = "image_loc.csv")
data_id = 2
print("***************************")
print(np.array(train_dataset[0][0][:,:,1])) #([data_id][image or label])
print(np.array(train_dataset[0][0].shape))  #[128 128  21]

dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
data = next(iter(dataloader))
x, t = data
print(t.shape)

# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = network.SNU_Network_classification(n_in=16384, n_mid=4096, n_out=2, num_time=10 ,gpu=True)
model = model.to(device)
print("building model")