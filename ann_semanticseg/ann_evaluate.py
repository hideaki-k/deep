import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import os 
from tqdm import tqdm
import sys
import cv2
import datetime

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
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        name = self.df['id'][i]
        #print('name',name)
        label = scipy.io.loadmat(self.df['label'][i])
        kyorigazou = cv2.imread(self.df['kyorigazou'][i])
        kyorigazou = cv2.cvtColor(kyorigazou, cv2.COLOR_BGR2GRAY)
        #image = image['time_data']
        label = label['label_data']
        kyorigazou = kyorigazou.reshape(4096)
        kyorigazou = kyorigazou.astype(np.float32)/255
        label = label.reshape(4096)
        label = label.astype(np.float32)
        return  label, kyorigazou, name
        

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
        self.conv3  = nn.Conv2d(in_channels = 4,
                               out_channels = 16,
                               kernel_size = 3,
                               padding = 1)

        self.conv4  = nn.Conv2d(in_channels = 16,
                        out_channels = 1,
                        kernel_size = 3,
                        padding = 1)   

        self.t_conv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16,
                                          kernel_size = 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1,
                                          kernel_size = 2, stride = 2)
        self.elu = nn.ELU()                                 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.up_samp = nn.Upsample(scale_factor=2,mode='nearest')

    def forward(self, x):

        x = x.squeeze(1)
        x = x.reshape((len(x), 1, 64, 64))
                
        #encode#                           
        x = self.elu(self.conv1(x))     
        x = self.pool(x)                  
        x = self.elu(self.conv2(x))      
        x = self.pool(x)                  
        #decode#
        x = self.elu(self.conv3(x)) 
        x = self.up_samp(x)   
        x = self.elu(self.conv4(x))
        x = self.up_samp(x)
        return x
        '''
        #encode#                           
        x = self.relu(self.conv1(x))     
        x = self.pool(x)                  
        x = self.relu(self.conv2(x))      
        x = self.pool(x)                  
        #decode#
        x = self.relu(self.t_conv1(x))    
        x = self.sigmoid(self.t_conv2(x))
        return x
        '''

def cal_average_iou(outputs, labels):

    outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
    labels = labels.data.cpu().numpy() 
    outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
    labels = labels.reshape(labels.shape[0],64,64)
    average_iou = 0
    average_recall = 0
    average_F = 0
    for i in range(30):
        iou_list = []
        precision_list = []
        recall_list = []
        F_list = []
        for j in range(0,10,1):
            j = j/10
            output = np.where(outputs[i]>j,1,0)
            label = np.where(labels[i]>0,1,0)
            intersection = (np.uint64(output) & np.uint64(label)).sum((0,1)) # will be zero if Trueth=0 or Prediction=0
            union = (np.uint64(output) | np.uint64(label)).sum((0,1)) # will be zero if both are 0
            smooth = 1e-6

            precision = (intersection / np.uint64(output).sum((0,1)))
            recall = (intersection / np.uint64(label).sum((0,1)))
            iou = ((intersection + smooth) / (union + smooth))
            F_score = 2*(precision*recall)/(recall+precision)
            F_score = round(F_score,5)
            iou = round(iou,5)
            precision = round(precision,5)
            recall = round(recall,5)
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)
            F_list.append(F_score)

        ind_iou = max(iou_list)
        average_iou += ind_iou
        ind = iou_list.index(ind_iou)
        average_recall += recall_list[ind]
        average_F += F_list[ind]
    print('average_iou is : ',average_iou/30) 
    print('average_recall is ',average_recall/30) 
    print('average_F is : ',average_F/30)  
    return average_iou/30

def iou_score(outputs, labels, data_id):
    smooth = 1e-6
    outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
    labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
    np.set_printoptions(threshold=np.inf)
    outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
    #labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
    labels = labels.reshape(labels.shape[0],64,64)
    print('outputs',outputs.shape)
    print('label',labels.shape)
    iou_list = []
    cnt = []
    precision_list = []
    recall_list = []
    F_list = []

    for j in range(0,10,1):
        j = j/10
        output = np.where(outputs[data_id]>j,1,0)
        label = np.where(labels[data_id]>0,1,0)
        intersection = (np.uint64(output) & np.uint64(label)).sum((0,1)) # will be zero if Trueth=0 or Prediction=0
        union = (np.uint64(output) | np.uint64(label)).sum((0,1)) # will be zero if both are 0

        iou = ((intersection + smooth) / (union + smooth))
        precision = (intersection / np.uint64(output).sum((0,1)))
        recall = (intersection / np.uint64(label).sum((0,1)))
        iou = ((intersection + smooth) / (union + smooth))
        F_score = 2*(precision*recall)/(recall+precision)
        F_score = round(F_score,5)
        iou = round(iou,5)
        precision = round(precision,5)
        recall = round(recall,5)
        #print('(j):IOU (',str(j)+') : '+str(iou))
        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)
        F_list.append(F_score)


    print('iou',iou_list)
    print('F_list',F_list)
    print('precision',precision_list)
    print('recall',recall_list)
    iou_max = max(iou_list)
    ind = iou_list.index(iou_max)
    print('max--iou',max(iou_list))
    print('max--precision',precision_list[ind])
    print('max--recall',recall_list[ind])
    print('max--F',F_list[ind])
    return iou_max,cnt,ind

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=50)
    args = parser.parse_args()


    print("***************************")
    valid_dataset = LoadDataset(r"C:\Users\aki\Documents\GitHub\deep\semantic_seg\evaluate_variety_dem_loc.csv")
    valid_iter = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False)
   

    # ネットワーク設計
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    #******ネットワークを選択******
    model = ConvAutoencoder().to(device)                           
    model_path = "0-5deg_simple(7680)/models_state_dict_end.pth"
    model.load_state_dict(torch.load(model_path))
    print("load model")

    ####　必ず確認！
    data_id = 4

    for i,(labels, kyorigazou, name) in enumerate(valid_iter, 0):
        if i== 19: #1-4 5-8 9-12 13-16
            break
        else:
            

            print('======i=======',i)
            inputs_ = kyorigazou
            label_ = labels
            inputs_ = inputs_.to(device)
            label_ = label_.to(device)
            pred = model(inputs_)

    print('inputs : ',inputs_.shape)
    print('pred :',pred.shape)
    print('label :',label_.shape)
    print('name is : ',name[data_id])
# pred:NN出力, label_:ラベル, inputs:NN入力
    device2 = torch.device('cpu')
    pred = pred.to(device2)

    ave_iou = cal_average_iou(pred, label_)
# IOU
    iou,_,max_iou_ind = iou_score(pred, label_, data_id)

    pred = pred.detach().clone().numpy()
    inputs_ = inputs_.to(device2)
    inputs_ = inputs_.detach().clone().numpy() 
    label_ = label_.to(device2)
    label_ = label_.detach().clone().numpy()
    #print('pred[data_id]',pred[data_id].shape) 
    pred = pred[data_id].squeeze(0)  #  (64, 64)
    label = label_[data_id].reshape(64,64) 
    print('label',label.shape)
    input = inputs_[data_id].reshape(64,64)
    print('input',input.shape)


    now = datetime.datetime.now()
    new_dir_path = 'log/'+now.strftime('%Y%m%d_%H%M%S')
    os.mkdir(new_dir_path)
    print("mkdir !")
    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111) 
    #ax.set_title('input')
    ax1.set_xlabel('x[pix]')
    ax1.set_ylabel('y[pix]')
    az = ax1.imshow(input,cmap=cm.winter)
    cbar = fig.colorbar(az,ticks=[0, 0.5, 1]) #, label='z[m]'
    cbar.ax.set_ylim(0,1)  
    cbar.ax.set_yticklabels(['Low', 'Medium', 'High']) 
    plt.savefig(str(new_dir_path)+"/DEM.png")

    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    ax1.set_title('label')
    ax1.imshow(label)   

    ax2 = fig.add_subplot(142) 
    ax2.set_title('input')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.imshow(input,cmap=cm.winter)

    ax3 = fig.add_subplot(143)
    ax3.set_title('output')    
    ax3.imshow(pred,cmap=cm.jet)
    
    max_iou_ind = max_iou_ind/10
    print("max_iou_ind",max_iou_ind)
    ax4 = fig.add_subplot(144)
    ax4.set_title('output_threshold')   
    ax4.set_xlabel('x[pix]')
    ax4.set_ylabel('y[pix]')
    pred_threshold = np.where(pred>max_iou_ind,1,0)
    ax4.imshow(pred_threshold,cmap=cm.gray)

    plt.tight_layout()
    OUT_FILE_NAME_ = str(new_dir_path)+"/log_image.png"
    plt.savefig(OUT_FILE_NAME_)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('ANN heatmap,iou is{}'.format((iou)))
    ax.imshow(pred,cmap=cm.jet)

    OUT_FILE_NAME_ = str(new_dir_path)+"/output_image.png"
    plt.savefig(OUT_FILE_NAME_)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('output_threshold')   
    ax.set_xlabel('x[pix]')
    ax.set_ylabel('y[pix]')
    pred_threshold = np.where(pred>max_iou_ind,1,0)
    ax.imshow(pred_threshold,cmap=cm.gray)
    plt.savefig(str(new_dir_path)+"/max_thresholding_image.png")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('error image')
    pred_threshold = np.where(pred_threshold>0, 0.5, 0)
    er_img = label + pred_threshold
    ax.imshow(er_img)
    OUT_FILE_NAME_ = str(new_dir_path)+"/error_image.png"
    plt.savefig(OUT_FILE_NAME_)



if __name__ == '__main__':
    main()

