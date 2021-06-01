import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io
#image = scipy.io.loadmat('crater/data_10.mat')
#image = scipy.io.loadmat('terrain_generation/perlin_bolder/data_10.mat')
#image = scipy.io.loadmat('two_craters_label/data_10.mat')
image = scipy.io.loadmat('64pix_two_craters_image/data_12.mat')
print(image)

print(type(image))
print(image.keys())
print(image.values())
print(image.items())
print(image['time_data'])

print(image['time_data'].shape)
#plt.imshow(image['label_data'])
#plt.show()

fig, ax = plt.subplots()
N = 10
def update(i):
    img = image['time_data'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=25)  # 代入しないと消される
plt.show()

label =  scipy.io.loadmat('64pix_two_craters_label/data_12.mat')
print(label.keys())
print(label.values())
print(label.items())
label_img = label['label_data']
plt.imshow(label_img)
plt.show()

""""
print(image['image'])
print(type(image['image']))
print(image['image'][:,:,1])
img = image['image'][:,:,1]
N = 100
fig, ax = plt.subplots()

def update(i):
    img = image['image'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=25)  # 代入しないと消される
plt.show()
"""