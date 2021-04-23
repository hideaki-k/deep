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
from torchsummary import summary


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
        #print("image : ",image['time_data'].shape)

        #label = torch.tensor(label, dtype=torch.float32)
        image = image['time_data']
        image = image.reshape(4096,21)
        #print("image : ",image.shape)
        image = image.astype(np.float32)
        label = label.astype(np.int64)
        label = torch.tensor(label,dtype =torch.int64 )
        #label = F.one_hot(label,num_classes=2)
        return image, label
     
 


train_dataset = TrainLoadDataset(csv_file = "image_loc.csv")
data_id = 2
print("***************************")
#print(np.array(train_dataset[0][0][:,:,1])) #([data_id][image or label])
#print(np.array(train_dataset[0][0].shape))  #[128 128  21]

train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)


# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = network.Conv_SNU_Network_classification(gpu=True)
model = model.to(device)
print("building model")
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 100
loss_hist = []
acc_hist = []
for epoch in range(epochs):
    running_loss = 0.0
    local_loss = []
    local_acc = []

    for i,(inputs, labels) in enumerate(train_iter, 0):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        #labels = labels.to(device,dtype=torch.long)
        #print("inputs:",inputs.shape) #torch.Size([128, 16384, 21])
        #print("labels:",labels.shape) #torch.Size([128])
        labels = labels.to(device)
        loss, pred, _ ,acc= model(inputs, labels)
        print(model)
        summary(model, [inputs, labels])

        pred,_ = torch.max(pred,1)
        #tmp = np.mean((_==labels).detach().cpu().numpy())
        #acc.append(tmp)  
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        local_loss.append(loss.item())
        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:.3f}'
                        .format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        local_acc.append(acc)
    #mean_acc = np.mean(acc)
    #acc_hist.append(mean_acc)
    mean_loss = np.mean(local_loss)
    loss_hist.append(mean_loss)
    mean_acc = np.mean(local_acc)
    acc_hist.append(mean_acc)
    print("EPOCH : ",epoch)
    print("mean_acc : ",mean_acc)

    
plt.figure(figsize=(3.3,2),dpi=150)
plt.plot(loss_hist)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
print("finish")

plt.figure(figsize=(3.3,2),dpi=150)
plt.plot(acc_hist)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
print(acc_hist)
print("finish")

model.eval()            #評価モード
inputs_, label_ = test_img()
inputs_ = inputs_.to(device)
label_ = label_.to(device)
print("inputs_",inputs_.shape)
print("label_",label_.shape)

label, pred, result = model(inputs_, label_)
print("result shape :",result.shape)

device2 = torch.device('cpu')
result = result.to(device2)
result = result.detach().clone().numpy()
# MP4  レコード
record(result)
