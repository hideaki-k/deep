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
from mp4_rec import mk_txt, record, rectangle_record, heatmap,label_save
import pandas as pd
import scipy.io
from torchsummary import summary
import argparse
import cv2

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
       
        self.df = pd.read_csv(csv_file)
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        name = self.df['id'][i]
        image = scipy.io.loadmat(self.df['id'][i])
        label = scipy.io.loadmat(self.df['label'][i])
        kyorigazou = cv2.imread(self.df['kyorigazou'][i])

        image = image['time_data']
        label = label['label_data']
        #print("image : ",image.shape)
        #print("label : ",label.shape)
        image = image.reshape(4096,20)
        #print("image : ",image.shape)
        image = image.astype(np.float32)
        #label = label.astype(np.int64)
        #label = torch.tensor(label,dtype =torch.int64 )
        label = label.reshape(4096)
        label = label.astype(np.float32)
        return image, label, kyorigazou, name
def iou_score(outputs, labels, data_id):
    smooth = 1e-6

    outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
    labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
    print('output',outputs.shape) #([128, 21, 1, 64, 64])

    outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
    #labels = labels.squeeze(1) #__labels.shape : (128, 64, 64)
    labels = labels.reshape(labels.shape[0],64,64)
    print('outputs',outputs.shape)
    print('label',labels.shape)
    iou_list = []
    cnt = []
    
    for j in range(0,20,1):
        output = np.where(outputs[data_id]>j,1,0)
        label = np.where(labels[data_id]>0,1,0)
        intersection = (np.uint64(output) & np.uint64(label)).sum((0,1)) # will be zero if Trueth=0 or Prediction=0
        union = (np.uint64(output) | np.uint64(label)).sum((0,1)) # will be zero if both are 0

        iou = ((intersection + smooth) / (union + smooth))
        #print('!!',iou.shape)
        iou = format(iou, '.5f').rstrip('0')
        print('(j):IOU (',str(j)+') : '+str(iou))
        iou_list.append(iou)
    
    iou_max = max(iou_list)
    return iou_max,cnt

parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=32)
parser.add_argument('--epoch', '-e', type=int, default=50)
parser.add_argument('--time', '-t', type=int, default=20,
                        help='Total simulation time steps.')
parser.add_argument('--rec', '-r', type=str, default=False)     
parser.add_argument('--forget', '-f', action='store_true' ,default=False)   
parser.add_argument('--dual', '-d', action='store_true' ,default=False)    
parser.add_argument('--power','-p', action='store_true', default=False)           
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     
#model = network.Fully_Connected_Gated_SNU_Net(num_time=args.time,l_tau=0.8,rec=args.rec, gpu=True,batch_size=args.batch)

# 全結合　畳み込みリカレントSNN
#model = network.Gated_CSNU_Net()
model = network.SNU_Network(rec=args.rec, forget=args.forget, dual=args.dual, power=args.power, batch_size=args.batch)
model = model.to(device)
model.eval()
print(model.state_dict().keys())
#model_path = "models/tof_input_DEM(16deg)/models_state_dict_end.pth"
#model_path = "models/recursive_tof_input(16deg)/models_state_dict_end.pth"
#model_path = "models/InputGated_recursive_tof(16deg)/models_state_dict_end.pth" 
#model_path = "models/dualGated_recurrent_tof(16deg)/models_state_dict_end.pth" # -d
model_path = "models/NC_KEN/0deg_SNU/models_state_dict_end.pth"
model.load_state_dict(torch.load(model_path))
print("load model")
#summary(model)

####　必ず確認！
data_id = 8
num_time = 20

# test_img : 評価用画像生成
# inputs_, label_ = test_img() 
#valid_dataset = LoadDataset("semantic_eval_loc.csv")
valid_dataset = LoadDataset("evaluate_variety_dem_loc.csv")
valid_iter = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False)
for i,(inputs, labels, kyorigazou, name) in enumerate(valid_iter, 0):
    if i==1:
        break
    else:
        print('======i=======',i)
        inputs_ = inputs
        label_ = labels
        inputs_ = inputs_.to(device)
        label_ = label_.to(device)
        #print("inputs_",inputs_.shape) # 
        #print("label_",label_.shape)

        if args.power:
            _, pred, result, _, _, spike_count = model(inputs_, label_)
            print("result shape : ",result.shape) #torch.Size([128, 21, 1, 64, 64])
            print("inputs_ shape : ",inputs_.shape) #torch.Size([128, 4096, 20])
            print('spike count:',spike_count)
            print('TOTAL SPIKE :',sum(spike_count))
        else:
            _, pred, result, _, _ = model(inputs_, label_)


print('name is :',name[data_id])
# IOU
iou,_ = iou_score(pred, label_, data_id)

device2 = torch.device('cpu')
result = result.to(device2)
result = result.detach().clone().numpy()
inputs_ = inputs_.to(device2)
inputs_ = inputs_.detach().clone().numpy() 
label_ = label_.to(device2)
label_ = label_.detach().clone().numpy() 

# MP4  レコード



#ディレクトリ生成
mk_txt(model_path,iou)
# label　の可視化
label_save(label_,data_id=data_id)
#results(最終層出力)の可視化
heatmap(result,kyorigazou, num_time=num_time,data_id=data_id,label_img=label_,iou=iou)

# inputs の可視化
rectangle_record(inputs_,num_time=num_time,data_id=data_id)
# result（最終層出力）の可視化
record(result, num_time=num_time,data_id=data_id)
