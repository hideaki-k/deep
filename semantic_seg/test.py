import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from cnn import cnn

if __name__ == '__main__':

    # GPU or CPUの自動判別                                                                 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # modelの定義                                                                          
    model = cnn().to(device)
    opt = torch.optim.Adam(model.parameters())

    # datasetの読み出し                                                                    
    bs = 128 # batch size                                                                  
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)

    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False)

    # training                                                                             
    print('train')
    model = model.train()
    for iepoch in range(3):
        for iiter, (x, y) in enumerate(trainloader, 0):

            # toGPU (CPUの場合はtoCPU)                                                     
            x = x.to(device)
            y = torch.eye(10)[y].to(device)

            # 推定                                                                         
            y_ = model.forward(x) # y_.shape = (bs, 84)                                    

            # loss: cross-entropy                                                          
            eps = 1e-7
            loss = -torch.mean(y*torch.log(y_+eps))

            opt.zero_grad() # 勾配初期化
            loss.backward() # backward (勾配計算)
            opt.step() # パラメータの微小移動

            # 100回に1回進捗を表示（なくてもよい）
            if iiter%100==0:
                print('%03d epoch, %05d, loss=%.5f' %
                      (iepoch, iiter, loss.item()))

    # test                                                                                 
    print('test')
    total, tp = 0, 0
    model = model.eval()
    for (x, label) in testloader:

        # to GPU                                                                           
        x = x.to(device)

        # 推定                                                                             
        y_ = model.forward(x)
        label_ = y_.argmax(1).to('cpu')

        # 結果集計                                                                         
        total += label.shape[0]
        tp += (label_==label).sum().item()

    acc = tp/total
    print('test accuracy = %.3f' % acc)

  