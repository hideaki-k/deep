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
from sklearn.metrics import roc_curve, auc
import csv

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, num_time):
        self.num_time = num_time
        self.df = pd.read_csv(csv_file)
        #self.data_transform = data_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image_name = self.df['id'][i]
        label_name = self.df['label'][i]
        image = scipy.io.loadmat(self.df['id'][i])
        label = scipy.io.loadmat(self.df['label'][i])
        kyorigazou = cv2.imread(self.df['kyorigazou'][i])
        
        image = image['time_data']
        label = label['label_data']
        #print("image : ",image.shape)
        #print("label : ",label.shape)
        image = image.reshape(4096,self.num_time)
        #print("image : ",image.shape)
        image = image.astype(np.float32)
        #label = label.astype(np.int64)
        #label = torch.tensor(label,dtype =torch.int64 )
        label = label.reshape(4096)
        label = label.astype(np.float32)
        return image, label, kyorigazou, image_name, label_name

def cal_average_iou(outputs, labels, num_time, batch_size):

    outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
    labels = labels.data.cpu().numpy() 
    outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
    labels = labels.reshape(labels.shape[0],64,64)
    average_iou = 0
    average_recall = 0
    average_precision = 0
    average_F = 0
    for i in range(batch_size):

        output = np.where(outputs[i]>=5,1,0)
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
        average_iou += iou
        average_precision += precision
        average_recall += recall


    print('average_iou is : ',average_iou/batch_size) 
    print('average_recall is ',average_recall/batch_size) 
    print('average_precision is : ',average_precision/batch_size)
    return average_iou/batch_size, average_precision/batch_size, average_recall/batch_size

def iou_score(outputs, labels, data_id, num_time):
    smooth = 1e-6

    outputs = outputs.data.cpu().numpy() #outputs.shape: (128, 1, 64, 64)
    labels = labels.data.cpu().numpy() #labels.shape: (128, 1, 64, 64)
    #print('output',outputs.shape) #([128, 21, 1, 64, 64])

    outputs = outputs.squeeze(1) # BATCH*1*H*W => BATCH*H*W __outputs.shape : (128, 64, 64)
    labels = labels.reshape(labels.shape[0],64,64)

    iou_list = []
    precision_list = []
    recall_list = []
    TPR_list = []
    FPR_list = []
    cnt = []
    F_list = []
    
    for j in range(0,num_time,1):
    # TP = intersection
    # TP + FN = label
    # TP + FP = output
    # FP + TN = 4096-label
   
        output = np.where(outputs[data_id]>j,1,0)
        label = np.where(labels[data_id]>0,1,0)
        intersection = (np.uint64(output) & np.uint64(label)).sum((0,1)) # will be zero if Trueth=0 or Prediction=0
        union = (np.uint64(output) | np.uint64(label)).sum((0,1)) # will be zero if both are 0
        
        precision = (intersection / np.uint64(output).sum((0,1)))
        recall = (intersection / np.uint64(label).sum((0,1)))
        iou = ((intersection + smooth) / (union + smooth))
        F_score = 2*(precision*recall)/(recall+precision)
        F_score = round(F_score,5)

        iou = round(iou,5)
        F_score = round(F_score,5)
        precision = round(precision,5)
        recall = round(recall,5)
        #print('(j):IOU (',str(j)+') : '+str(iou))
        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)
        F_list.append(F_score)
        #TPR_list.append(TPR)
        #FPR_list.append(FPR)

    print('iou',iou_list)
    print('F_list',F_list)
    print('precision',precision_list)
    print('recall',recall_list)

    # const_threshold iouのマックスのj（最終層スパイク数)を固定
    const_threshold = 1
    const_val = 4
    if const_threshold:
        iou_max = iou_list[const_val]
        ind = const_val
        print('5--iou',iou_list[const_val])
        print('5--precision',precision_list[const_val])
        print('5--recall',recall_list[const_val] )
        print('5--F',F_list[const_val])
    else:
        iou_max = max(iou_list)
        ind = iou_list.index(iou_max)
        print('max--iou',max(iou_list))
        print('max--precision',precision_list[ind])
        print('max--recall',recall_list[ind])
        print('max--F',F_list[ind])

    return iou_max,  cnt, ind

parser = argparse.ArgumentParser()
parser.add_argument('--batch', '-b', type=int, default=32)
parser.add_argument('--epoch', '-e', type=int, default=50)
parser.add_argument('--time', '-t', type=int, default=11,
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
model = network.SNU_Network(num_time=args.time, rec=args.rec, forget=args.forget, dual=args.dual, power=args.power, batch_size=args.batch)
model = model.to(device)
model.eval()
print(model.state_dict().keys())
#model_path = "models/tof_input_DEM(16deg)/models_state_dict_end.pth"
#model_path = "models/recursive_tof_input(16deg)/models_state_dict_end.pth"
#model_path = "models/InputGated_recursive_tof(16deg)/models_state_dict_end.pth" 
#model_path = "models/dualGated_recurrent_tof(16deg)/models_state_dict_end.pth" # -d
if args.dual:
    model_path = "models/master_thethis/0-3deg_rsnu(t-10)_lidar_noised_boulder/models_state_dict_end.pth"
else:
    model_path = "models/master_thethis/0-3deg_snu(t-10)_lidar_noised_boulder/models_state_dict_end.pth"
#model_path = "models/ists/0deg_rSNU/models_state_dict_end.pth"
model.load_state_dict(torch.load(model_path))
print("load model")
summary(model)

####　必ず確認！
data_id = 9
num_time = args.time
# パラメータ書き込み
iou_log_csv = 0

# test_img : 評価用画像生成
# inputs_, label_ = test_img() 

valid_dataset = LoadDataset("master_evaluation.csv", num_time)
#valid_dataset = LoadDataset("semantic_eval_loc.csv", num_time)
#valid_dataset = LoadDataset("evaluate_1124.csv", num_time)
#valid_dataset = LoadDataset("evaluate_variety_dem_loc.csv", num_time)
valid_iter = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False)
for i,(inputs, labels, kyorigazou, name, label_name) in enumerate(valid_iter, 0):
   #1-4 5-8 9-12 13-16 17-20(batch=32)
   #1-16 17-33 34-50 51-67 68-84(batch=8)
        print('======i=======',i)
        inputs_ = inputs
        label_ = labels
        inputs_ = inputs_.to(device)
        label_ = label_.to(device)
        depth = kyorigazou
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
        
        if iou_log_csv:
            ave_iou,ave_precision,ave_recall = cal_average_iou(pred, label_, num_time=num_time, batch_size=args.batch)
            with open('log_all_parameters.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([str(i),str(ave_iou),str(ave_precision),str(ave_recall)])


        elif i== 9: 
            break

#for data_id in range(args.batch):
    #print('name is :',name[data_id])
    # IOU
ave_iou,_,_ = cal_average_iou(pred, label_, num_time=num_time, batch_size=args.batch)
iou,_,max_iou_ind = iou_score(pred, label_, data_id, num_time=num_time)
iou = str(iou)
print('name is :',name[data_id])
print('label is : ',label_name[data_id])
device2 = torch.device('cpu')
result = result.to(device2)
result = result.detach().clone().numpy()
inputs_ = inputs_.to(device2)
inputs_ = inputs_.detach().clone().numpy() 
label_ = label_.to(device2)
label_ = label_.detach().clone().numpy() 
# MP4  レコード



#ディレクトリ生成
mk_txt(model_path,iou,name[data_id])
# label　の可視化
label_save(label_,data_id=data_id)
#results(最終層出力)の可視化
heatmap(result, depth, num_time=num_time, data_id=data_id,label_img=label_,iou=iou,max_iou_ind=max_iou_ind)

# inputs の可視化
rectangle_record(inputs_,num_time=num_time,data_id=data_id)
# result（最終層出力）の可視化
record(result, num_time=num_time,data_id=data_id)
