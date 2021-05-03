import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io

def rectangle(img, img_ano, centers, max_side):
    """
    img 四角形の線のみの二次元画像
    img_ano アノテーション画像
    centers center座標のlist
    max_side 辺の最大長さの1/2
    """
    if max_side < 3: # max_side が小さすぎるとき
        max_side = 4
    # 辺の長さの1/2を定義
    side_x = np.random.randint(3, int(max_side))
    side_y = np.random.randint(3, int(max_side))

    # 中心の座標(x, y)を定義
    x = np.random.randint(max_side + 1, img.shape[0] - (max_side + 1))
    y = np.random.randint(max_side + 1, img.shape[1] - (max_side + 1))

    #過去の中心位置と近い位置が含まれた場合,inputデータをそのまま返す
    for center in centers:
        if np.abs(center[0] - x ) < (2 * max_side + 1):
            if np.abs(center[1] - y) < (2 * max_side + 1):
                return img, img_ano, centers
    
    img[x - side_x : x + side_x, y - side_y] = 1.0      #上辺
    img[x - side_x : x + side_x, y + side_y] = 1.0      #下辺
    img[x - side_x, y - side_y : y + side_y] = 1.0      #左辺
    img[x + side_x, y - side_y : y + side_y + 1] = 1.0  #右辺
    img_ano[x - side_x : x + side_x + 1, y - side_y : y + side_y + 1] = 1.0
    centers.append([x, y])
    return img, img_ano, centers


def test_img(num_images=128, length=64, num_time=20, N=256, max_fr=300, dt = 1e-3):
    #img_test = np.zeros([num_images, 1, length, length])
    #imgs_test_ano = np.zeros([num_images, 1, length, length])
    imgs = np.zeros([num_images, length, length]) #ゼロ行列を生成、入力画像
    imgs_ano = np.zeros([num_images, length, length]) # 出力画像

    for i in range(num_images):
        centers = []
        img = np.zeros([length, length])
        img_ano = np.zeros([length, length])
        for j in range(3):
            img, img_ano, centers = rectangle(img, img_ano, centers, 7)
        #img_test[i, 0, :, :] = img
        imgs[i, :, :] = img
        imgs_ano[i, :, :] = img_ano

    imgs = torch.tensor(imgs, dtype=torch.float32)    # imgs = torch.Size([1000, 1, 64, 64]) -> imgs :  torch.Size([1000, 64, 64])
    imgs_ano = torch.tensor(imgs_ano, dtype=torch.float32)


    imgs = imgs.reshape(imgs.shape[0],-1)/255
    imgs_ano = imgs_ano.reshape(imgs_ano.shape[0],-1)/255


    #data_set = TensorDataset(imgs, imgs_ano)
    #data_loader = DataLoader(data_set, batch_size = 256, shuffle = True)

    #print("imgs : ", imgs.shape) # imgs = torch.Size([1000, 1, 64, 64])
    #print("data_set : ",data_set)

    """
    imgs_binary = np.heaviside(imgs[0],0)
    plt.imshow(imgs_binary.reshape(64,64))
    plt.show()
    """
    x = np.zeros((num_images,4096,num_time))
    y = np.zeros((num_images,4096))

    for i in tqdm(range(N)):
        fr = max_fr * np.repeat(np.expand_dims(np.heaviside(imgs[i],[0]),1),num_time,axis=1)
        x[i] = np.where(np.random.rand(4096, num_time) < fr*dt, 1, 0)
        y[i] = imgs_ano[i]

    data_id = 0
    print("x shape:",x.shape)
    print(np.array(x[data_id].shape)) #[4096  100]
    sum = np.sum(x[data_id],axis=1)
    """
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(np.reshape(sum,(64,64)))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(np.reshape(x[data_id][:,0],(64,64)))
    plt.show()
    """
    x = torch.from_numpy(x.astype(np.float32)).clone()
    y = torch.from_numpy(y.astype(np.float32)).clone()
    return x, y

def make_img(num_images=256, length=64, num_time=20, N=256, max_fr=300, dt = 1e-3):
    
    imgs = np.zeros([num_images, length, length]) #ゼロ行列を生成、入力画像
    imgs_ano = np.zeros([num_images, length, length]) # 出力画像

    for i in range(num_images):
        centers = []
        img = np.zeros([length, length])
        img_ano = np.zeros([length, length])
        for j in range(3):
            img, img_ano, centers = rectangle(img, img_ano, centers, 7)
        #img_test[i, 0, :, :] = img
        imgs[i, :, :] = img
        imgs_ano[i, :, :] = img_ano

    imgs = torch.tensor(imgs, dtype=torch.float32)    # imgs = torch.Size([1000, 1, 64, 64]) -> imgs :  torch.Size([1000, 64, 64])
    imgs_ano = torch.tensor(imgs_ano, dtype=torch.float32)


    imgs = imgs.reshape(imgs.shape[0],-1)/255
    imgs_ano = imgs_ano.reshape(imgs_ano.shape[0],-1)/255


    #data_set = TensorDataset(imgs, imgs_ano)
    #data_loader = DataLoader(data_set, batch_size = 256, shuffle = True)

    #print("imgs : ", imgs.shape) # imgs = torch.Size([1000, 1, 64, 64])
    #print("data_set : ",data_set)

    """
    imgs_binary = np.heaviside(imgs[0],0)
    plt.imshow(imgs_binary.reshape(64,64))
    plt.show()
    """
    x = np.zeros((num_images,4096,num_time))
    y = np.zeros((num_images,4096))

    for i in tqdm(range(num_images)):
        fr = max_fr * np.repeat(np.expand_dims(np.heaviside(imgs[i],[0]),1),num_time,axis=1)
        x[i] = np.where(np.random.rand(4096, num_time) < fr*dt, 1, 0)
        y[i] = imgs_ano[i]

    data_id = 0
    print("x shape:",x.shape)
    print(np.array(x[data_id].shape)) #[4096  100]
    for data_id in range(num_images):
        print(data_id)
        img = x[data_id].reshape(64,64,x.shape[2])
        label = y[data_id].reshape(64,64)
        scipy.io.savemat("semantic_data/image/img_"+str(data_id)+".mat",{"name":img})
        scipy.io.savemat("semantic_data/label/label_"+str(data_id)+".mat",{"name":label})

if __name__ == '__main__':

    make_img(num_images=25600, length=64, num_time=20, N=256, max_fr=300, dt = 1e-3)
    """
    inputs, label = test_img(num_images=12800)
    print("inputs shape : ",inputs.shape)
    print("label shape : ",label.shape)
    # mat保存
    scipy.io.savemat("rect_inputs.mat", {'name':inputs})
    scipy.io.savemat("rect_label.mat", {'name':label})
    """