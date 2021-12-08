import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data.dataloader
from torch.utils.data import DataLoader

import DataProcess
from DataLoad import GasDataset


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1))

    def forward(self, input):
        return self.main(input)


def train():
    dataSet = GasDataset()
    dataloader = DataLoader(dataSet, 10, shuffle=False)

    net = Net(367)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 随机梯度下降算法
    loss_func = torch.nn.MSELoss()
    lossList = []
    for i in range(10):
        loss = 0
        print('epoch ', i)
        for gas, label in dataloader:
            out = net.forward(gas)
            loss = loss_func(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossList.append(loss.item())
            print( '\ny = ', label[0].item(), 'predict = ', out[0].item(), '\nloss = ', loss.item())
        print('finish one epoch loss = ', loss.item())

    plt.plot(lossList)


if __name__ == '__main__':
    train()
