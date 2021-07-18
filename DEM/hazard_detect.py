import glob
import scipy.io
import numpy as  np
import matplotlib.pyplot as plt

files = glob.glob("128pix_(0deg)_dem/model/*.mat")
m = scipy.io.loadmat(files[0])['model'].shape[0]
n = scipy.io.loadmat(files[0])['model'].shape[1]
print(m,n)
V = np.zeros(m,n)
S = np.zeros(m,n)
R = np.zeros(m,n)

for file in files:

    mat = scipy.io.loadmat(file)['model']

    plt.imshow(mat)
    plt.show()
