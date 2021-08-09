import glob
import scipy.io
import numpy as  np
import matplotlib.pyplot as plt
import cv2
import os

new_dir_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)/hazard_label'
os.makedirs(new_dir_path, exist_ok=True)
files = glob.glob(r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)/model/*.mat')

rei = scipy.io.loadmat(files[1])['DEM']
print(rei)
height = rei.shape[0]
width = rei.shape[1]
# ウィンドウ大きさ
F = 8


def Get_Slope(h, w, D, mu, sigma):

    roi = D[(h-F//2):(h+F//2),(w-F//2):(w+F//2)]
    mu_roi = np.mean(roi)
    suiheido = (mu_roi - mu)/sigma
    return abs(suiheido)

def Get_Roughness(h, w, D, mu, sigma):

    roi = D[(h-F//2):(h+F//2),(w-F//2):(w+F//2)]
    s_roi = np.std(roi)
    heitando = s_roi/sigma
    return heitando

print(height,width)


for file in files:
    print('input : ',file)
    target = '\model_'
    idx = file.find(target)
    cnt = file[idx+7:]
    idx = cnt.find('.mat')
    cnt = cnt[:idx]
    DEM = scipy.io.loadmat(file)['DEM'] 

    mu = np.mean(DEM)
    sigma = np.std(DEM)
    print("heikin ",mu)

 
    laplacian =  cv2.Laplacian(DEM,cv2.CV_64F)
    V = np.zeros((height,width)) # safety for each pixel
    S = np.zeros((height,width)) # slope for each pixel
    R = np.zeros((height,width)) # roughness for each pixel
    
    for row in range(F//2, height-(F//2), 1):
        for col in range(F//2, width-(F//2), 1):
            suiheido = Get_Slope(row, col, DEM, mu, sigma)
            heitando = Get_Roughness(row, col, DEM, mu, sigma)
            if suiheido > S[row][col]:
                S[row][col] = suiheido
            if heitando > R[row][col]:
                R[row][col] = heitando
    
    Vthm = np.mean(S)
    Vths = np.mean(R)
    S = S>Vthm
    R = R>1.5*Vths
    hazard = (S|R)

    save_path = new_dir_path + '/label_'+ cnt + '.mat'
    print('output : ',save_path)
    scipy.io.savemat(save_path, {'label_data':hazard})

    #np.set_printoptions(threshold=np.inf)
    #print(S+R)
 
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,4,1)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)
    ax1.imshow(DEM)
    ax2.set_title('suiheido')
    ax2.imshow(S) 
    ax3.imshow(hazard)
    ax4.set_title('heitando')
    ax4.imshow(R)
    fig.savefig('figure01.jpg')
    plt.show()
  
    

    
