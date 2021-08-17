import glob
import scipy.io
import numpy as  np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import os
from ransac import *
from sklearn.preprocessing import MinMaxScaler

new_dir_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(5deg)_dem(noisy)/hazard_label'
os.makedirs(new_dir_path, exist_ok=True)
files = glob.glob(r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(5deg)_dem(noisy)/model/*.mat')

rei = scipy.io.loadmat(files[1])['true_DEM']
print(rei)
height = rei.shape[0]
width = rei.shape[1]
# ウィンドウ大きさ
F = 8

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[:10, :10]
    return xx, yy, (-d - a * xx - b * yy) / c
    
def Get_Slope(h, w, D, mu, sigma):
    #fig = plt.figure()
    #ax = mplot3d.Axes3D(fig)
    
    n = 100
    max_iterations = 100
    goal_inliers = n * 0.3

    xyzs = np.zeros((64, 3))

    roi = D[(h-F//2):(h+F//2),(w-F//2):(w+F//2)]
    #print(roi.shape)
    #print(roi)
    i = 0
    for x in range(8):
        for y in range(8):
    
            z = roi[x][y]
            #print(i,x,y,z)
            xyzs[i][0] = x 
            xyzs[i][1] = y
            xyzs[i][2] = z
            i += 1
    #print(xyzs)
    #ax.scatter3D(xyzs.T[0], xyzs.T[1], xyzs.T[2])
    # RANSAC
    m, best_inliers = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    #ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))
    #plt.show()
    ans = math.degrees(math.atan(abs(a/c)) + math.degrees(math.atan(abs(b/c))))
    print(ans)
    return ans

    

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
    DEM = scipy.io.loadmat(file)['true_DEM'] 

    mu = np.mean(DEM)
    sigma = np.std(DEM)
    print("heikin ",mu)

 
    V = np.zeros((height,width)) # safety for each pixel
    S = np.zeros((height,width)) # slope for each pixel
    R = np.zeros((height,width)) # roughness for each pixel
    
    for row in range(F//2, height-(F//2), 1):
        for col in range(F//2, width-(F//2), 1):
            suiheido = Get_Slope(row, col, DEM, mu, sigma)
            print("suiheido:",suiheido)
            heitando = Get_Roughness(row, col, DEM, mu, sigma)
            S[row][col] = suiheido
            print("heitando",heitando)
 

            if heitando > R[row][col]:
                R[row][col] = heitando
    S = min_max(S)
    np.set_printoptions(threshold=np.inf)
    print("S:",S)
    print("R:",R)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax1.set_title('original')
    ax1.imshow(DEM)
    ax2.set_title('suihei')
    ax2.imshow(S) 
    ax3.set_title('heitan')
    ax3.imshow(R)
    plt.show()
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
  
    

    
