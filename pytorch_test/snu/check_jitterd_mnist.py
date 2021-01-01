import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

# Here we load the Dataset
root = os.path.expanduser("/fashion-mnist")
train_dataset = torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.MNIST(root, train=False, transform=None, target_transform=None, download=True)

# Standardize data
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=np.float)
print(len(x_train))
x_train = x_train.reshape(x_train.shape[0],-1)/255
print(x_train[2])

x_train = x_train[2]
x_binary = np.heaviside(x_train,0)
print(x_binary[2])

num_time = 20
fr = 100 # Hz
dt = 1e-3

x_binary_fr = fr * np.repeat(np.expand_dims(x_binary,1),num_time,axis=1)
x = np.where(np.random.rand(784, num_time) < x_binary_fr*dt,1,0)
sum_x = np.sum(x,axis=1)
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0],-1)/255

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=np.int)
y_test  = np.array(test_dataset.targets, dtype=np.int)
data_id = 2

## animation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
OUT_FILE_NAME = "output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
dst = cv2.imread('image.png') #一度、savefigしてから再読み込み
rows,cols,channels = 28,28,3
out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
fig = plt.figure()

ims = []
for i in range(num_time):
  print(x[:,i].reshape(28,28).shape)
  im=plt.imshow(x[:,i].reshape(28,28), cmap=plt.cm.gray_r,animated=True)
  ims.append([im])
  #plt.show("im",im)
  print(ims)
  ani = animation.ArtistAnimation(fig, ims, interval=100)
  fig1=plt.pause(0.001)
  #Gifアニメーションのために画像をためます
  plt.savefig(str(i)+".png")
  dst = cv2.imread(str(i)+'.png')
  out.write(dst) #mp4やaviに出力します
  print(i)




fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(x_train.reshape(28,28), cmap=plt.cm.gray_r)
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(x_binary.reshape(28,28), cmap=plt.cm.gray_r)
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(np.reshape(sum_x,(28,28)))
plt.show()
