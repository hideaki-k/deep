import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os 
import datetime

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
        print("x[data_id] shape",x[data_id].shape)
        print(x[data_id][:,i].reshape(64,64).shape)
        im=plt.imshow(x[data_id][:,i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
        ims.append([im])
        #plt.show()
        print(ims)
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        fig1=plt.pause(0.001)
        #Gifアニメーションのために画像をためます
        plt.savefig(str(new_dir_path)+"/original_image_"+str(i)+".png")
        dst = cv2.imread(str(new_dir_path)+"/original_image_"+str(i)+'.png')
        out.write(dst) #mp4やaviに出力します
        print(i)
    out.release()


OUT_FILE_NAME = str(new_dir_path)+"/result_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def record(x,num_time=10,data_id=2):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img_4_25/image.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
    fig = plt.figure()
    ims = []

    for i in range(num_time):
        print(x[data_id][i].reshape(64,64))
        print("sum:",sum(x[data_id][i]))
        im=plt.imshow(x[data_id][i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
        ims.append([im])
        #plt.show()
        print(ims)
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        fig1=plt.pause(0.001)

        #spk = x[data_id][i].reshape(64,64)
        plt.savefig(str(new_dir_path)+"/result_image_"+str(i)+'.png')
        dst = cv2.imread(str(new_dir_path)+"/result_image_"+str(i)+'.png')
        out.write(dst) #mp4やaviに出力します
        print(i)
    out.release()

if __name__ == '__main__':
    OUT_FILE_NAME = "result_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img_4_25/result_image4.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
    #fig = plt.figure()
    ims = []
    for i in range(20):
        dst = cv2.imread("20210503_155548/result_image"+str(i)+".png")
        #cv2.imshow('sam',dst)
        #cv2.show()
        #cv2.waitKey(0)
        #cv2.destroyAllwindows()

        out.write(dst) #mp4やaviに出力します
        print(i)
        