import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import os 
from tqdm import tqdm
import sys
import cv2


from tqdm import tqdm
#from mp4_rec import record, rectangle_record
import pandas as pd
import scipy.io
from torchsummary import summary
import argparse
from PIL import Image

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.image_paths = self.df['id']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        
        image = cv2.imread(self.image_paths[i] ) # 画像読み取り
        label = scipy.io.loadmat(self.df['label'][i]) #matfile　kuga_label or alhat_label 

        label = label['label_data']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        np.set_printoptions(threshold=np.inf)
        #print("image",image) # 0 ~ 255
        image = image.reshape(4096)
        #print("image : ",image.shape) # (4096,)
       
        image = image.astype(np.float32)/255
        
        #label = label.astype(np.int64)
        #label = torch.tensor(label,dtype =torch.int64 )
        label = label.reshape(4096)
        label = label.astype(np.float32)
        

        #print('label',label.shape) # (4096,) 0 or 1
        return image, label

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = 3,
                               padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16,
                               out_channels = 4,
                               kernel_size = 3,
                               padding = 1)
        #Decoder Layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16,
                                          kernel_size = 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1,
                                          kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.squeeze(1)
        x = x.reshape((len(x), 1, 64, 64))
        
        #encode#                           
        x = self.relu(self.conv1(x))     
        x = self.pool(x)                  
        x = self.relu(self.conv2(x))      
        x = self.pool(x)                  
        #decode#
        x = self.relu(self.t_conv1(x))    
        x = self.sigmoid(self.t_conv2(x))
        return x

def iou_score(outputs, labels):
    smooth = 1e-6
    outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
    labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
    np.set_printoptions(threshold=np.inf)
    outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
    labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
    #print("outputs : ",outputs)
    iou = []
    cnt = []
    
    output = np.where(outputs>0.5,1,0)
    label = np.where(labels>0,1,0)
    intersection = (np.uint64(output) & np.uint64(label)).sum((1,2)) # will be zero if Trueth=0 or Prediction=0
    union = (np.uint64(output) | np.uint64(label)).sum((1,2)) # will be zero if both are 0

    iou.append((intersection + smooth) / (union + smooth))
    iou = np.mean(iou)
    
    return iou,cnt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    args = parser.parse_args()


    print("***************************")
    train_dataset = LoadDataset("ann_img_loc.csv")
    test_dataset = LoadDataset("ann_eval_loc.csv")
    data_id = 2
    print(np.array(train_dataset[data_id][0]).shape) #(784, 100) 
    train_iter = DataLoader(train_dataset, batch_size=args.batch, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # ネットワーク設計
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    #******ネットワークを選択******
    net = ConvAutoencoder().to(device)                           
    loss_fn = nn.MSELoss()                                #損失関数の定義
    optimizer = optim.Adam(net.parameters(), lr = 1e-4)
    acces = []
    losses = []                                     #epoch毎のlossを記録
    epoch_time = 30
    for epoch in range(epoch_time):
        running_loss = 0.0 
        running_iou = 0.0                         #epoch毎のlossの計算
        net.train()
        for i, (inputs, labels) in enumerate(train_iter):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #labels= torch.where(labels>0,255,0).to(torch.float32)
            optimizer.zero_grad()       
            y_pred = net(inputs)
            torch.set_printoptions(threshold=100000)
            #print('y_pred',y_pred)
            #print('labels',labels.shape)
            labels = labels.reshape((len(labels), 1, 64, 64))
            loss = loss_fn(y_pred, labels)
            #### IOU
            iou,cnt= iou_score(y_pred, labels)
            #print('iou',iou)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_iou += iou
        print("epoch:",epoch, " loss : iou ", running_loss/(i + 1),running_iou/(i + 1))

        losses.append(running_loss/(i + 1))
        acces.append(running_iou/(i+1))

    #lossの可視化
    fig = plt.figure(facecolor='oldlace')
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(losses)
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch time")

    ax2.plot(acces)
    ax2.set_ylabel("IOU")
    ax2.set_xlabel("epoch time")
    plt.savefig("loss_iou")
    plt.show()
if __name__ == '__main__':
    main()
