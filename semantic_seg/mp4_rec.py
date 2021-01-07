import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


# rectangle_record ・・学習データ可視化関数
# record ・・　出力データ可視化関数

OUT_FILE_NAME_ = "build_img/original_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def rectangle_record(x,num_time=10,data_id=2):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img/image.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME_, int(fourcc), int(10), (int(cols), int(rows)))
    fig = plt.figure()
    ims = []

    for i in range(num_time):
        print(x[data_id][:,i].reshape(64,64).shape)
        im=plt.imshow(x[data_id][:,i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
        ims.append([im])
        #plt.show()
        print(ims)
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        fig1=plt.pause(0.001)
        #Gifアニメーションのために画像をためます
        plt.savefig("build_img/original_image_"+str(i)+".png")
        dst = cv2.imread("build_img/original_image_"+str(i)+'.png')
        out.write(dst) #mp4やaviに出力します
        print(i)


OUT_FILE_NAME = "build_img/result_output_video.mp4"
#OUT_FILE_NAME = "output_video.avi"
def record(x,num_time=10,data_id=2):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img/image.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
    fig = plt.figure()
    ims = []

    for i in range(num_time):
        print(x[data_id][i].reshape(64,64))
        print("sum:",sum(x[data_id][i]))
        im=plt.imshow(x[data_id][i].reshape(64,64), cmap=plt.cm.gray_r,animated=True)
        ims.append([im])
        plt.show()
        print(ims)
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        fig1=plt.pause(0.1)
        #Gifアニメーションのために画像をためます
        #plt.savefig("build_img/result_image"+str(i)+".png")

        spk = x[data_id][i].reshape(64,64)
        plt.imsave("build_img/result_image"+str(i)+'.png',spk)
        dst = cv2.imread("build_img/result_image"+str(i)+'.png')
        out.write(dst) #mp4やaviに出力します
        print(i)


if __name__ == '__main__':
    OUT_FILE_NAME = "build_img/result_output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst = cv2.imread('build_img/result_image_ver2_5.png') #一度、savefigしてから再読み込み
    rows,cols,channels = dst.shape
    out = cv2.VideoWriter(OUT_FILE_NAME, int(fourcc), int(10), (int(cols), int(rows)))
    #fig = plt.figure()
    ims = []
    for i in range(10):
        dst = cv2.imread("build_img/result_image"+str(i)+".png")
        #cv2.imshow('sam',dst)
        #cv2.show()
        #cv2.waitKey(0)
        #cv2.destroyAllwindows()

        out.write(dst) #mp4やaviに出力します
        print(i)
        