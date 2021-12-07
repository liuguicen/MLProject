import math

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class GasDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10000

    def __getitem__(self, id):
        x = torch.randn(1) * 100
        return x, torch.as_tensor([my_fun(x[0])], dtype=torch.float)


def my_fun(x):
    return 0.7 * math.cos(x) + 0.5 * math.sin(x) + math.sin(x) * math.cos(x) + math.sin(0.1 * x)


def test(net):
    sample_x = 0
    realx, realy, pre_y = [], [], []
    while sample_x < 10:
        realx.append(sample_x)
        realy.append(my_fun(sample_x))
        sample_x += 0.2
    plt.plot(realx, realy)
    npx = np.array(realx)
    print(npx)
    tensorx = torch.from_numpy(npx).unsqueeze(1).float()
    pre_y = net.forward(tensorx)
    pre_y = pre_y.detach().squeeze().numpy()
    print("real y", realy)
    print("pre_y ", pre_y)
    plt.plot(realx, pre_y)
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seuential = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, input):
        return self.seuential(input)


model = Net()
batch_size = 100


def train():
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()  # 均方误差，用于计算预测值与真实值之间的误差
    loss_list = []
    dataloader = torch.utils.data.dataloader.DataLoader(GasDataset(), batch_size=100)
    for input, real_y in dataloader:
        optimizer.zero_grad()
        # print(input, real_y)
        pre_y = model.forward(input)
        # print(pre_y.size())
        # print(real_y.size(), pre_y.size())
        loss = criterion(pre_y, real_y)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        loss_list.append(loss.item())
    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    train()
    test(model)
