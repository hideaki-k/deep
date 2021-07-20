import glob
import scipy.io
import numpy as  np
import matplotlib.pyplot as plt
import cv2

files = glob.glob("128pix_(0deg)_dem/model/*.mat")
height = scipy.io.loadmat(files[0])['model'].shape[0]
width = scipy.io.loadmat(files[0])['model'].shape[1]
F = 8


def Get_Slope(h, w, D):
    roi = D[(h-F//2):(h+F//2),(w-F//2):(w+F//2)]
    laplacian =  cv2.Laplacian(D,cv2.CV_64F)
    slope = np.amax(roi) - np.amin(roi)

    return slope

def Get_Roughness(h, w, D):
    roi = D[(h-F//2):(h+F//2),(w-F//2):(w+F//2)]
    return np.mean(roi)

print(height,width)
V = np.zeros((height,width)) # safety for each pixel
S = np.zeros((height,width)) # slope for each pixel
R = np.zeros((height,width)) # roughness for each pixel

for file in files:
    DEM = scipy.io.loadmat(file)['model']
    laplacian =  cv2.Laplacian(DEM,cv2.CV_64F)
    #lt.imshow(laplacian)
    #plt.show()
    


    for row in range(F//2, height-(F//2), 1):
        for col in range(F//2, width-(F//2), 1):
            s = Get_Slope(row, col, DEM)
            r = Get_Roughness(row, col, DEM)
            if s > S[row][col]:
                S[row][col] = s
            if r > R[row][col]:
                R[row][col] = r
    print(S)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.imshow(DEM)
    ax2.imshow(S) 
    ax3.imshow(S>3)

    plt.show()

    
