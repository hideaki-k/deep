import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/content/drive/My Drive/Colab_Notebooks/Pytorch_test/SNU_PyTorch")
from model import snu_layer
from model import network
from tqdm import tqdm

class TrainLoadDataset(torch.utils.data.Dataset):
  def __init__(self, N=60000, dt=1e-3, num_time=100, max_fr=60):
    root = os.path.expanduser("mnist")
    train = torchvision.datasets.MNIST(root,train=True, transform=None, target_transform=None, download=True)
    label = np.array(train.targets, dtype=np.int)
    print("label : ",label.shape)
    #label = label.type(torch.LongTensor)


    x = np.zeros((N, 784, num_time)) # 784 = 28 * 28
    y = np.zeros(N)
    train = np.array(train.data, dtype=np.float)
    train = train.reshape(train.shape[0],-1)/255
    

    train_binary = np.heaviside(train[2],0)
    #plt.imshow(train_binary.reshape(28,28))
    for i in tqdm(range(N)):
      fr = max_fr * np.repeat(np.expand_dims(np.heaviside(train[i],[0]),1),num_time,axis=1)
      x[i] = np.where(np.random.rand(784, num_time) < fr*dt, 1, 0)
      y[i] = label[i]
    self.x = x.astype(np.float32)
    self.y = y.astype(np.int8)
    self.N = N

  def __len__(self):
    return self.N

  def __getitem__(self, i):
    return self.x[i], self.y[i]
  
class TestLoadDataset(torch.utils.data.Dataset):
  def __init__(self, N=3000, dt=1e-3, num_time=100, max_fr=60):
    root = os.path.expanduser("~/data/datasets/torch/mnist/test")
    test = torchvision.datasets.MNIST(root,train=False, transform=None, target_transform=None, download=True)
    label = np.array(test.targets, dtype=np.int)
    x = np.zeros((N, 784, num_time)) # 784 = 28 * 28
    y = np.zeros(N)
    test = np.array(test.data, dtype=np.float)
    test = test.reshape(test.shape[0],-1)/255
    

    test_binary = np.heaviside(test[2],0)
    #plt.imshow(train_binary.reshape(28,28))
    for i in tqdm(range(N)):
      fr = max_fr * np.repeat(np.expand_dims(np.heaviside(test[i],[0]),1),num_time,axis=1)
      x[i] = np.where(np.random.rand(784, num_time) < fr*dt, 1, 0)
      y[i] = label[i]
    self.x_ = x.astype(np.float32)
    self.y_ = y.astype(np.int8)
    self.N_ = N

  def __len__(self):
    return self.N_

  def __getitem__(self, i):
    return self.x_[i], self.y_[i]

def main():
  train_dataset = TrainLoadDataset(N=25600, dt=1e-3, num_time=100, max_fr=700)
  test_dataset = TestLoadDataset(N=2560, dt=1e-3, num_time=100, max_fr=700)
  # plot debig
  data_id = 2
  print(np.array(train_dataset[data_id][0]).shape)
  sum = np.sum(train_dataset[data_id][0],axis=1)
  """
  fig = plt.figure(figsize=(6,3))
  ax1 = fig.add_subplot(1,2,1)
  ax1.imshow(np.reshape(sum,(28,28)))
  ax2 = fig.add_subplot(1,2,2)
  ax2.imshow(np.reshape(train_dataset[data_id][0][:,0],(28,28)))
  plt.show()
  """

  train_iter = DataLoader(train_dataset, batch_size=256, shuffle=True)
  test_iter = DataLoader(test_dataset, batch_size=256, shuffle=True)
  #plt.imshow(np.reshape(dataset[1][0][:,0],(28,28)))
  print("building model")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = network.SNU_Network(n_in=784, n_mid=256, n_out=10, num_time=10, gpu=True)
  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-4)
  epochs = 100
  loss_hist = []
  acc_hist = []
  for epoch in range(epochs):
    running_loss = 0.0
    local_loss = []
    acc = []
    for i,(inputs, labels) in enumerate(train_iter, 0):
      # zero the paramerter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      inputs = inputs.to(device)
      labels = labels.to(device,dtype=torch.int64)
      #print("inputs shape:",inputs.shape)
      #print("inputs shape:",inputs.shape)
      #print("labels shape:",labels)
      #print("type labels:",labels.device)
      loss , pred= model(inputs,labels)
      print("loss:",loss)
      print("pred :",pred.shape)
      print("labels:",labels.shape)

      pred,_ = torch.max(pred,1)
      
      print("_ : ",_)
      tmp = np.mean((_==labels).detach().cpu().numpy())
      acc.append(tmp)
      
      


      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      local_loss.append(loss.item())
      if i % 100 == 99:
        print('[{:d}, {:5d}] loss: {:.3f}'
                    .format(epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0
    mean_acc = np.mean(acc)
    mean_loss = np.mean(local_loss)
    loss_hist.append(mean_loss)
    acc_hist.append(mean_acc)
    
  plt.figure(figsize=(3.3,2),dpi=150)
  plt.plot(loss_hist)
  plt.xlabel("epoch")
  plt.ylabel("Loss")
  plt.show()
  print("finish")
  
  plt.figure(figsize=(3.3,2),dpi=150)
  plt.plot(acc_hist)
  plt.xlabel("epoch")
  plt.ylabel("acc")
  plt.show()
  print("finish")

if __name__ == '__main__':
    main()


