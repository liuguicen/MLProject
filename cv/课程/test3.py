import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataset import Dataset


# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # torch.Size([100, 1]) #把[a,b,c]变成[[a,b,c]]
# print(x)
# y = 2 * (x.pow(2)) + 0.5 * torch.rand(x.size())  # torch.rand为均匀分布，返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
# print(y)
# 画图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

def my_fun(x):
    return 100 * (0.7 * math.cos(x) + 0.5 * math.sin(x) + math.sin(x) * math.cos(x) + math.sin(0.1 * x))


class GasDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, id):
        x = torch.randn(1)
        # return x, x[0] * 2 + x[1] * 3)
        return x, torch.as_tensor([my_fun(x[0] * 10)], dtype=torch.float)


class NetWork(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(NetWork, self).__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.hidden1 = nn.Linear(n_hidden, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        self.output_for_predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))  # 对x进入隐层后的输出应用激活函数（相当于一个筛选的过程）
        x = F.leaky_relu(self.hidden1(x))
        x = F.leaky_relu(self.hidden2(x))
        output = self.output_for_predict(x)  # 做线性变换，将维度为1
        return output


def test(net):
    net.eval()
    sample_x = 0
    realx, realy, pre_y = [], [], []
    while sample_x < 11:
        realx.append(sample_x)
        realy.append(my_fun(sample_x))
        sample_x += 0.2
    plt.plot(realx, realy)
    npx = np.array(realx)
    npx /= 10
    print(npx)
    tensorx = torch.from_numpy(npx).unsqueeze(1).float()
    pre_y = net.forward(tensorx)
    print("real y", realy)
    print("pre y tensor", pre_y)
    pre_y = pre_y.detach().squeeze().numpy()
    print("pre_y ", pre_y)
    plt.plot(realx, pre_y)
    plt.show()


network = NetWork(n_input=1, n_hidden=256, n_output=1)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # 均方误差，用于计算预测值与真实值之间的误差
dataloader = torch.utils.data.dataloader.DataLoader(GasDataset(), batch_size=200)

epoch = 200
loss_list = []
number = 1
for i in range(epoch):  # 训练步数（相当于迭代次数）
    realy = None
    realx = None
    predication = None
    for x, y in dataloader:
        realx = x
        predication = network(x)
        loss = criterion(predication, y)  # predication为预测的值，y为真实值
        # if i == epoch - 1:
        # print(y, predication)
        realy = y
        optimizer.zero_grad()
        loss.backward()  # 反向传播，更新参数
        optimizer.step()  # 将更新的参数值放进network的parameters

        print(loss.item())
        if number % 100 == 0:
            loss_list.append(loss.item())
        number += 1

    if (i == epoch - 1):
        realx = realx.squeeze().detach().numpy()
        realy = realy.squeeze().detach().numpy()
        predication = predication.squeeze().detach().numpy()
        Z = zip(realx, realy)  # 对AB进行封装，把频率放在前面
        Z = sorted(Z)  # 进行逆序排列
        B, A = zip(*Z)  # 进行解压，其中的AB已经按照频率排好
        plt.plot(B, A)
        Z = zip(realx, predication)  # 对AB进行封装，把频率放在前面
        Z = sorted(Z)  # 进行逆序排列
        C, D = zip(*Z)  # 进行解压，其中的AB已经按照频率排好
        plt.plot(C, D)
        plt.show()

plt.plot(loss_list)
plt.show()

test(network)