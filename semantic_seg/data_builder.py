import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

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

num_images = 1000 #生成する画像数
length = 64       #画像のサイズ
imgs = np.zeros([num_images, 1, length, length]) #ゼロ行列を生成、入力画像
imgs_ano = np.zeros([num_images, 1, length, length]) # 出力画像

for i in range(num_images):
    centers = []
    img = np.zeros([64, 64])
    img_ano = np.zeros([64, 64])
    for j in range(6):
        img, img_ano, centers = rectangle(img, img_ano, centers, 12)
    #plt.imshow(img_ano)
    #plt.show()
    imgs[i, 0, :, :] = img
    imgs_ano[i, 0, :, :] = img_ano

imgs = torch.tensor(imgs, dtype=torch.float32)
imgs_ano = torch.tensor(imgs_ano, dtype=torch.float32)
data_set = TensorDataset(imgs, imgs_ano)
data_loader = DataLoader(data_set, batch_size = 100, shuffle = True)

print("imgs : ", imgs.shape)
print("data_set : ",data_set)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        #Encoder Layers
        self.conv1 = nn.Conv2d(in_channels = 1,
                               out_channels = 16,
                               kernel_size = 3,
                               padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16,
                               out_channels = 4,
                               kernel_size = 3,
                               padding = 1)
        #Decoder Layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16,
                                          kernel_size = 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1,
                                          kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #encode#                           
        x = self.relu(self.conv1(x))        
        x = self.pool(x)                  
        x = self.relu(self.conv2(x))      
        x = self.pool(x)                  
        #decode#
        x = self.relu(self.t_conv1(x))    
        x = self.sigmoid(self.t_conv2(x))
        return x

#******ネットワークを選択******
net = ConvAutoencoder()
loss_fn =nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.01)

losses = []
epoch_time = 30

for epoch in range(epoch_time):
    running_loss = 0.0
    net.train()
    for i, (XX, yy) in enumerate(data_loader):
        optimizer.zero_grad()
        y_pred = net(XX)
        loss = loss_fn(y_pred, yy)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("epoch:","loss:",running_loss/(i + 1))
    losses.append(running_loss/(i + 1))
#lossの可視化
plt.plot(losses)
plt.ylabel("loss")
plt.xlabel("epoch time")
plt.savefig("loss_auto")
plt.show()

net.eval() # 評価モード
#今まで学習していない画像を一つ生成
num_images = 1
img_test = np.zeros([num_images, 1, length, length])
imgs_test_ano = np.zeros([num_images, 1, length, length])
for i in range(num_images):
    centers = []
    img = np.zeros([length, length])
    img_ano = np.zeros([length, length])
    for j in range(6):
        img, img_ano, centers = rectangle(img, img_ano, centers,7)
    img_test[i, 0, :, :,] = img

img_test = img_test.reshape([1, 1, 64, 64])
img_test = torch.tensor(img_test, dtype=torch.float32)
img_test = net(img_test)
img_test = img_test.detach().numpy()
img_test = img_test[0, 0, :, :]

plt.imshow(img, cmap="gray") # input データの可視化
plt.savefig("input_auto")
plt.show()
plt.imshow(img_test, cmap="gray") #outputデータの可視化
plt.show()
plt.imshow(img_ano, cmap="gray") # 正解データの可視化
plt.savefig("correct_auto") # 正解データ
plt.plot()