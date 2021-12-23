import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import cv2
import os 
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib._version import get_versions as mplv
from scipy.stats import gaussian_kde
# rectangle_record ・・学習データ可視化関数
# record ・・　出力データ可視化関数
now = datetime.datetime.now()
new_dir_path = 'log/'+now.strftime('%Y%m%d_%H%M%S')
os.mkdir(new_dir_path)
print("mkdir !")

OUT_FILE_NAME_ = str(new_dir_path)+"/inputs_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def mk_txt(model_name,iou,name):
    model_name = model_name.replace('models/','')
    model_name = model_name.replace('/','__')
    model_name = model_name.replace('.pth','')
    path = str(new_dir_path)+"/"+str(model_name)+'.txt'
    f = open(path,'w')
    f.write(iou)
    f.write('\n')
    f.write(name)
    f.close()

def rectangle_record(x,num_time=20,data_id=2):

    fig, ax = plt.subplots()
    N = 20
    def update(i):
        if i == 1:
            print("===========")
        img = x[data_id][:,i].reshape(64,64)
        #print(i)
        
        plt.clf()
        
        plt.imshow(img,cmap='gray')
    ani = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=1)  # 代入しないと消される

    ani.save(str(new_dir_path)+"/inputs.gif", writer="pillow", fps=10)
    #plt.show()

def label_save(label,data_id):
    print("label shape",label.shape)
    image = label[data_id].reshape(64,64)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x[pix]')
    ax.set_ylabel('y[pix]')
    ax.imshow(image,cmap=cm.gray)
    fig.savefig(str(new_dir_path)+'/label.png')


OUT_FILE_NAME = str(new_dir_path)+"/result_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def record(x,num_time=20,data_id=2): # x : output
    print(x[2][1].shape)
    print(x[2][1][:].shape)
    fig, ax = plt.subplots()
    N = 20
    def update(i):
        if i == 1:
            print("===========")
        img = x[data_id][i].reshape(64,64)
        #print(i)
        
        plt.clf()
        
        plt.imshow(img,cmap='gray')
    ani = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=1)  # 代入しないと消される

    ani.save(str(new_dir_path)+"/outputs.gif", writer="pillow", fps=10)
    #plt.show()

def heatmap(x,kyorigazou,num_time=20,data_id=2,label_img=None, iou=0,max_iou_ind=7): # 画像を時間軸方向に積算してヒートマップに
    print("x",x.shape)
    x = x[data_id].squeeze(1)
    kyorigazou = kyorigazou[data_id]
    #x = x[data_id]
    print("x",x.shape) # x (11, 64, 64)
    sum_x = np.sum(x,axis=0)
    print("sum_x",sum_x.shape) # sum_x (64, 64)

    fig = plt.figure()
    print("max_iou_ind",max_iou_ind)
    ax = fig.add_subplot(111)
    pred_threshold = np.where(sum_x > max_iou_ind, 1, 0)
    ax.set_xlabel('x[pix]')
    ax.set_ylabel('y[pix]')
    ax.imshow(pred_threshold,cmap=cm.gray)
    plt.savefig(str(new_dir_path)+"/max_thresholding_image.png")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #H = ax.hist2d(sum_x[0],sum_x[1],bins=40,cmap=cm.jet)
    ax.set_title('heat map,iou is{}'.format((iou)))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.imshow(sum_x,cmap=cm.jet)
    
    #plt.show()  
    plt.savefig(str(new_dir_path)+"/heatmap_image.png")
    #print("save heat")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x[pix]')
    ax.set_ylabel('y[pix]')
    ax.imshow(kyorigazou,cmap=cm.winter)
    plt.savefig(str(new_dir_path)+"/DEM.png")
 

    # visualize IoU
    label_img = label_img[data_id].reshape(64,64)
    
    l = np.zeros((sum_x.shape[0],sum_x.shape[1]))
    
    ref_img_2 = np.where(sum_x > 1, 0.5, 0)
    ref_img_3 = np.where(sum_x > 9, 0.5, 0)
    ref_img_4 = np.where(sum_x > 18 , 0.5, 0)
    IoU_img_2 = label_img + ref_img_2
    IoU_img_3 = label_img + ref_img_3
    IoU_img_4 = label_img + ref_img_4

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('error image')
    ax.imshow(IoU_img_3)
    OUT_FILE_NAME_ = str(new_dir_path)+"/error_image.png"
    plt.savefig(OUT_FILE_NAME_)

    fig = plt.figure(facecolor='azure')
    ax1 = fig.add_subplot(2,4,1)
    ax2 = fig.add_subplot(2,4,2)
    ax3 = fig.add_subplot(2,4,3)
    ax4 = fig.add_subplot(2,4,4)
    ax_ = fig.add_subplot(2,4,5)
    ax5 = fig.add_subplot(2,4,6)
    ax6 = fig.add_subplot(2,4,7)
    ax7 = fig.add_subplot(2,4,8)


    ax1.set_title('label')
    ax1.imshow(label_img,cmap='bwr')

    ax2.set_title('output(1)')
    ax2.imshow(ref_img_2,cmap='bwr')

    ax3.set_title('output(9)')
    ax3.imshow(ref_img_3,cmap='bwr')

    ax4.set_title('output(18)')
    ax4.imshow(ref_img_4,cmap='bwr')

    ax_.set_title('kyorigazou')
    ax_.imshow(kyorigazou)

    ax5.set_title('IOU image(1)')
    ax5.imshow(IoU_img_2)

    ax6.set_title('IoU image(9)')
    ax6.imshow(IoU_img_3)

    ax7.set_title('Iou image(18)')
    ax7.imshow(IoU_img_4)

    plt.tight_layout()
    plt.savefig(str(new_dir_path)+"/IoU_image.png")
    


if __name__ == '__main__':
    ###ヒートマップとして出力


    ###動画出力
    OUT_FILE_NAME = "result_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img_4_25/result_image4.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
    #fig = plt.figure()
    ims = []
    for i in range(20):
        dst = cv2.imread("20210507_092631/result_image"+str(i)+".png")
        #cv2.imshow('sam',dst)
        #cv2.show()
        #cv2.waitKey(0)
        #cv2.destroyAllwindows()

        out.write(dst) #mp4やaviに出力します
        print(i)
        