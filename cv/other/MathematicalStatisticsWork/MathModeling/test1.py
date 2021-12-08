import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # torch.Size([100, 1]) #把[a,b,c]变成[[a,b,c]]
# print(x)
y = 2 * (x.pow(2)) + 0.5 * torch.rand(x.size())  # torch.rand为均匀分布，返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
# print(y)
# 画图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

from torch import nn
import torch.nn.functional as F


class GasDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, id):
        x = torch.unsqueeze(torch.linspace(-1, 1, 20), dim=1)  # torch.Size([100, 1]) #把[a,b,c]变成[[a,b,c]]
        # print(x)
        y = 2 * (x.pow(2)) + 0.5 * torch.rand(
            x.size())  # torch.rand为均匀分布，返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义
        return x, y


class NetWork(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(NetWork, self).__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.hidden1 = nn.Linear(n_hidden, n_hidden)
        self.output_for_predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))  # 对x进入隐层后的输出应用激活函数（相当于一个筛选的过程）
        x = F.relu(self.hidden1(x))
        output = self.output_for_predict(x)  # 做线性变换，将维度为1
        return output


network = NetWork(n_input=1, n_hidden=8, n_output=1)
print(network)  # 打印模型的层次结构

plt.ion()  # 打开交互模式
plt.show()

optimizer = torch.optim.SGD(network.parameters(), lr=0.2)
criterion = torch.nn.MSELoss()  # 均方误差，用于计算预测值与真实值之间的误差
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # torch.Size([100, 1]) #把[a,b,c]变成[[a,b,c]]
# print(x)
y = 2 * (x.pow(2)) + 0.5 * torch.rand(x.size())  # torch.rand为均匀分布，返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义

dataloader = torch.utils.data.dataloader.DataLoader(GasDataset(), batch_size=20)
for i in range(200):  # 训练步数（相当于迭代次数）
    predication = network(x)
    loss = criterion(predication, y)  # predication为预测的值，y为真实值

    optimizer.zero_grad()
    loss.backward()  # 反向传播，更新参数
    optimizer.step()  # 将更新的参数值放进network的parameters
    print(loss.item())
    if i % 5 == 0:
        plt.cla()  # 清坐标轴
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predication.data.numpy(), 'ro', lw=5)  # 画预测曲线，用红色o作为标记
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
