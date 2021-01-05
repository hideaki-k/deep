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
from mp4_rec import record
class TrainLoadDataset(torch.utils.data.Dataset):
    def __init__(self, N=1000, dt=1e-3, num_time=100, max_fr=300):
        num_images = N #生成する画像数
        length = 64       #画像のサイズ
        
        self.N = N
        num_time = num_time 
        max_fr = max_fr
        dt = dt
        #imgs = np.zeros([num_images, 1, length, length]) #ゼロ行列を生成、入力画像
        #imgs_ano = np.zeros([num_images, 1, length, length]) # 出力画像
        imgs = np.zeros([num_images, length, length]) #ゼロ行列を生成、入力画像
        imgs_ano = np.zeros([num_images, length, length]) # 出力画像

        # rectangle 生成
        for i in range(num_images):
            centers = []
            img = np.zeros([64, 64])
            img_ano = np.zeros([64, 64])
            for j in range(6):
                img, img_ano, centers = rectangle(img, img_ano, centers, 12)
            #plt.imshow(img_ano)
            #plt.show()
            imgs[i, :, :] = img
            imgs_ano[i, :, :] = img_ano


        imgs = torch.tensor(imgs, dtype=torch.float32)    # imgs = torch.Size([1000, 1, 64, 64]) -> imgs :  torch.Size([1000, 64, 64])
        imgs_ano = torch.tensor(imgs_ano, dtype=torch.float32)


        imgs = imgs.reshape(imgs.shape[0],-1)/255
        imgs_ano = imgs_ano.reshape(imgs_ano.shape[0],-1)/255


        data_set = TensorDataset(imgs, imgs_ano)
        data_loader = DataLoader(data_set, batch_size = 256, shuffle = True)

        print("imgs : ", imgs.shape) # imgs = torch.Size([1000, 1, 64, 64])
        print("data_set : ",data_set)


        imgs_binary = np.heaviside(imgs[2],0)
        plt.imshow(imgs_binary.reshape(64,64))
        plt.show()
        x = np.zeros((N,4096,num_time))
        y = np.zeros((N,4096))


        for i in tqdm(range(N)):
            fr = max_fr * np.repeat(np.expand_dims(np.heaviside(imgs[i],[0]),1),num_time,axis=1)
            x[i] = np.where(np.random.rand(4096, num_time) < fr*dt, 1, 0)
            y[i] = imgs_ano[i]

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

        data_id = 2
        print("self.x shape:",self.x.shape)
        print(np.array(self.x[data_id].shape)) #[4096  100]
        sum = np.sum(self.x[data_id],axis=1)

        fig = plt.figure(figsize=(6,3))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(np.reshape(sum,(64,64)))
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(np.reshape(self.x[data_id][:,0],(64,64)))
        plt.show()

    
    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return self.x[i], self.y[i]    


train_dataset = TrainLoadDataset(N=2560, dt=1e-3, num_time=10, max_fr=300)
data_id = 2
print("***************************==============")
print(np.array(train_dataset[data_id][0]).shape) #(784, 100) 
train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)

# ネットワーク設計
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = network.SNU_Network(n_in=4096, n_mid=4096, n_out=4096, num_time=10, gpu=True)
model = model.to(device)
print("building model")
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 100
loss_hist = []
acc_hist = []
for epoch in range(epochs):
    running_loss = 0.0
    local_loss = []
    acc = []

    for i,(inputs, labels) in enumerate(train_iter, 0):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        #labels = labels.to(device,dtype=torch.long)
        labels = labels.to(device)
        loss, pred, _ = model(inputs, labels)
        print("loss : ",loss)
        print("pred :",pred.shape)
        print("labels:",labels.shape)

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
    #mean_acc = np.mean(acc)
    #acc_hist.append(mean_acc)
    mean_loss = np.mean(local_loss)
    loss_hist.append(mean_loss)

    
plt.figure(figsize=(3.3,2),dpi=150)
plt.plot(loss_hist)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
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



"""
plt.figure(figsize=(3.3,2),dpi=150)
plt.plot(acc_hist)
plt.xlabel("epoch")
plt.ylabel("acc")
plt.show()
print("finish")
"""





"""動画生成

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
OUT_FILE_NAME = "build_img/output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
dst = cv2.imread('build_img/image.png') #一度、savefigしてから再読み込み
rows,cols,channels = dst.shape
out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
fig = plt.figure()
ims = []

for i in range(num_time):
  print(x[data_id][:,i].reshape(64,64).shape)
  im=plt.imshow(x[data_id][:,i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
  ims.append([im])
  #plt.show("im",im)
  print(ims)
  ani = animation.ArtistAnimation(fig, ims, interval=100)
  fig1=plt.pause(0.001)
  #Gifアニメーションのために画像をためます
  plt.savefig("build_img/image"+str(i)+".png")
  dst = cv2.imread("build_img/image"+str(i)+'.png')
  out.write(dst) #mp4やaviに出力します
  print(i)

"""

