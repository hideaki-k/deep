import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io
#time_data = scipy.io.loadmat('crater/data_10.mat')
#time_data = scipy.io.loadmat('terrain_generation/perlin_bolder/data_10.mat')
time_data = scipy.io.loadmat('two_craters_label/data_10.mat')
print(time_data)

print(type(time_data))
print(time_data.keys())
print(time_data.values())
print(time_data.items())
print(time_data['label_data'])

print(time_data['label_data'].shape)
plt.imshow(time_data['label_data'])
plt.show()


""""
print(time_data['time_data'])
print(type(time_data['time_data']))
print(time_data['time_data'][:,:,1])
img = time_data['time_data'][:,:,1]
N = 100
fig, ax = plt.subplots()

def update(i):
    img = time_data['time_data'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=25)  # 代入しないと消される
plt.show()
"""