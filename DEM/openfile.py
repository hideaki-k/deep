import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io
time_data = scipy.io.loadmat('crater/data_5.mat')
print(time_data)

print(type(time_data))
print(time_data.keys())
print(time_data.values())
print(time_data.items())
print(time_data['time_data'])
print(type(time_data['time_data']))
print(time_data['time_data'][:,:,1])
img = time_data['time_data'][:,:,1]
N = 20
fig, ax = plt.subplots()
def update(i):
    img = time_data['time_data'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=25)  # 代入しないと消される
plt.show()