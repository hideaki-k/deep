import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
sys.path.append(r'C:\Users\aki\Documents\GitHub\deep\pytorch_test\Surrogate_BP')
from model import unet_model
from torch import optim
from tqdm import tqdm
import pandas as pd
import scipy.io
import logging

class CSVLoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
       
        image = scipy.io.loadmat(self.df['id'][i])
        label = scipy.io.loadmat(self.df['label'][i])

        #print(image.shape)
        #print(label.shape)
        image = image['time_data'] #(262144,20)
        label = label['label_data'] #(512, 512)
        print(image.shape)
        image = image.reshape(64,64,11)
        image = image.astype(np.float32)
        label = label.reshape(64,64)
        label = label.astype(np.float32)
        return image, label

def train_net(model,batch_size,device):
    train_dataset = CSVLoadDataset('UNet_image_loc.csv')
    # train_dataset [data_id],[0]:時系列データ,[1]:ラベルデータ 
    # data_id = 1
    # print(np.array(train_dataset[data_id][0].shape))
    # print(np.array(train_dataset[data_id][1].shape))
    # plt.imshow(train_dataset[data_id][1])
    # plt.show()
    print("batch", batch_size)
    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    # ネットワーク設計
    device = device
    net = model
    net.to(device=device)
    print("building network")
    print(net)
    n_train = len(train_dataset)
    optimizer = optim.Adam(net.parameters(),lr=1e-5)
    criterion = nn.MSELoss()
    epochs = 10
    loss_hist = []
    acc_hist = []

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        local_loss = []
        local_acc = []
        with tqdm(total=n_train,desc=f'Epoch {epoch+1}/{epochs}',unit='img') as pbar:
            for i,(inputs,labels) in enumerate(train_iter, 0):
                inputs = inputs.to(device=device,dtype=torch.float32) #([128, 512, 512, 21])
                labels = labels.to(device=device,dtype=torch.float32) #([128, 512, 512])
                #print('inputs.shape',inputs.shape)
                pred = net(inputs)
                """
                pred = pred.to('cpu').detach().numpy().copy()
                print("pred shape",pred[0,0].shape)
                plt.imshow(pred[0,0])
                plt.show()
                """

                loss = criterion(pred, labels)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss(batch)':loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(inputs.shape[0])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = unet_model.UNet(n_channels=1,n_classes=1,num_time=10,gpu=True)
    print("start")
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    try:
        train_net(net,batch_size=2,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(),'INTERRUPTED.pth')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)