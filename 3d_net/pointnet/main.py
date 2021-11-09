from model import *
from sampler import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
num_points = 64
num_labels = 1

def main():
    pointnet = PointNet(num_points, num_labels)

    new_param = pointnet.state_dict()
    new_param['main.0.main.6.bias'] = torch.eye(3,3).view(-1)
    new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
    pointnet.load_state_dict(new_param)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(pointnet.parameters(),lr=0.001)

    loss_list = []
    accuracy_list = []

    for iteration in range(100+1):

        pointnet.zero_grad()

        input_data, labels = data_sampler(batch_size, num_points)
        print('input data',input_data.shape)
        output = pointnet(input_data)
        output = nn.Sigmoid()(output)

        error = criterion(output, labels)
        error.backward()

        optimizer.step()

        with torch.no_grad():
            output[output>0.5] = 1
            output[output<0.5] = 0
            accuracy = (output==labels).sum().item()/batch_size
        
        loss_list.append(error.item())
        accuracy_list.append(accuracy)

        if iteration % 10 == 0:
            print('Iteration : {}   Loss : {}'.format(iteration, error.item()))
            print('Iteration : {}   Accuracy : {}'.format(iteration, accuracy))

    plt.plot(accuracy_list)
    plt.show()
if __name__ == '__main__':
    main()
