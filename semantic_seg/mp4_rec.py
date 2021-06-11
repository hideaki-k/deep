import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import cv2
import os 
import datetime
import torch
# rectangle_record ・・学習データ可視化関数
# record ・・　出力データ可視化関数
now = datetime.datetime.now()
new_dir_path = 'log/'+now.strftime('%Y%m%d_%H%M%S')
os.mkdir(new_dir_path)
print("mkdir !")

OUT_FILE_NAME_ = str(new_dir_path)+"/inputs_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def mk_txt(model_name):
    model_name = model_name.replace('models/','')
    model_name = model_name.replace('.pth','')
    path = str(new_dir_path)+"/"+str(model_name)+'.txt'
    f = open(path,'w')
    f.write(model_name)
    f.close()

def rectangle_record(x,num_time=10,data_id=2):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img_4_25/image.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME_, int(fourcc), int(10), (int(cols), int(rows)))
    fig = plt.figure()
    ims = []

    for i in range(num_time):
        print("inputs_[data_id] shape",x[data_id].shape)
        print(x[data_id][:,i].reshape(64,64).shape)
        im=plt.imshow(x[data_id][:,i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
        ims.append([im])
        #plt.show()
        #print(ims)
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        fig1=plt.pause(0.01)
        #Gifアニメーションのために画像をためます
        plt.savefig(str(new_dir_path)+"/original_image_"+str(i)+".png")
        dst = cv2.imread(str(new_dir_path)+"/original_image_"+str(i)+'.png')
        out.write(dst) #mp4やaviに出力します
        #print(i)
    out.release()

def label_save(label,data_id):
    print("label shape",label.shape)
    image = label[data_id].reshape(64,64)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    fig.savefig(str(new_dir_path)+'/label.png')


OUT_FILE_NAME = str(new_dir_path)+"/result_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def record(x,num_time=10,data_id=2): # x : output
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img_4_25/image.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
    fig = plt.figure()
    ims = []
    print("result shape",x.shape)

    for i in range(num_time):
        #print(x[data_id][i].reshape(64,64))
        #print("sum:",sum(x[data_id][i]))
        im=plt.imshow(x[data_id][i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
        ims.append([im])
        #plt.show()
        #print(ims)
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        fig1=plt.pause(0.01)

        #spk = x[data_id][i].reshape(64,64)
        plt.savefig(str(new_dir_path)+"/result_image_"+str(i)+'.png')
        dst = cv2.imread(str(new_dir_path)+"/result_image_"+str(i)+'.png')
        out.write(dst) #mp4やaviに出力します
        #print(i)
    out.release()

def heatmap(x,num_time=10,data_id=2,label_img=None): # 画像を時間軸方向に積算してヒートマップに
    x = x[data_id].squeeze(1)
    print("x",x.shape) # x (11, 64, 64)
    sum_x = np.sum(x,axis=0)
    print("sum_x",sum_x.shape) # sum_x (64, 64)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #H = ax.hist2d(sum_x[0],sum_x[1],bins=40,cmap=cm.jet)
    ax.set_title('heat map')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.imshow(sum_x,cmap=cm.jet)
    
    #plt.show()  
    plt.savefig(str(new_dir_path)+"/heatmap_image.png")
    print("save heat")

    # visualize IoU
    label_img = label_img[data_id].reshape(64,64)
    
    l = np.zeros((sum_x.shape[0],sum_x.shape[1]))
    
    ref_img_5 = np.where(sum_x > 5 , 1, 0)
    ref_img_3 = np.where(sum_x>3,1,0)
    IoU_img = label_img + ref_img_5
    IoU_img_ = label_img + ref_img_3
    fig = plt.figure(facecolor='azure')
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
   
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)

    ax1.set_title('laabel')
    ax1.imshow(label_img)

    ax2.set_title('output')
    ax2.imshow(ref_img_3)

    ax3.set_title('IoU image(5)')
    ax3.imshow(IoU_img)

    ax4.set_title('IoU image(3)')
    ax4.imshow(IoU_img_)
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
        