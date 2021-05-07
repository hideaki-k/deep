import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io

"""
#### label imageを見たいときはコメントアウト外してください

name = scipy.io.loadmat('label/label_100.mat')
print(name)
print(name['name'].shape)

img = name['name']
np.set_printoptions(threshold=np.inf)
print(img)

plt.imshow(img)
plt.show()
"""
##### image　を見たいときはコメントアウトを外してください

name = scipy.io.loadmat('image/img_100.mat')
print(name)

print(type(name))
print(name.keys())
print(name.values())
print(name.items())
print(name['name'])
print(type(name['name']))
print("============")
print(name['name'].shape)
print(name['name'][:,:,1])
img = name['name'][:,:,1]
np.set_printoptions(threshold=np.inf)
print(img)
"""
N = 100
fig, ax = plt.subplots()
def update(i):
    img = name['name'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=25)  # 代入しないと消される
plt.show()
"""