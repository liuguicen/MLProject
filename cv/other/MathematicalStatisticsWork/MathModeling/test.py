import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

import DataLoad


class GasDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 2000

    def __getitem__(self, id):
        x = torch.randn(1)
        # return x, x[0] * 2 + x[1] * 3)
        return x, torch.as_tensor([x[0] * 2], dtype=torch.float)


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.f1 = nn.Linear(input_dim, input_dim * 8)
        self.f2 = nn.Linear(input_dim * 8, input_dim * 4)
        self.f3 = nn.Linear(input_dim * 4, input_dim * 2)
        self.f4 = nn.Linear(input_dim * 2, input_dim // 2)
        self.f5 = nn.Linear(input_dim // 2, input_dim // 2)
        self.f6 = nn.Linear(input_dim // 2, 1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.f1(x))
        x = nn.functional.leaky_relu(self.f2(x))
        x = nn.functional.leaky_relu(self.f3(x))
        x = nn.functional.leaky_relu(self.f4(x))
        x = nn.functional.leaky_relu(self.f5(x))
        x = self.f6(x)
        return x


network = Net(DataLoad.input_dim)
optimizer = torch.optim.Adam(network.parameters(), lr=0.0002)
# optimizer = torch.optim.SGD(network.parameters(), lr=0.000001)
criterion = torch.nn.MSELoss()
dataloader = DataLoader(DataLoad.GasDataset(), batch_size=100, shuffle=True)

lossList = []
for i in range(150):
    loss = 0
    for x, y in dataloader:
        predication = network(x)
        loss = criterion(predication, y)  # predication为预测的值，y为真实值

        optimizer.zero_grad()
        loss.backward()  # 反向传播，更新参数
        optimizer.step()  # 将更新的参数值放进network的parameters

        para = network.named_parameters()
        # for name, p in para:
        #     print(name, p)
        print('y = ', y[0].item(), 'predict = ', predication[0].item(), '\n', loss.item())
    lossList.append(loss.item())

plt.plot(lossList)
plt.show()
    # if i % 10 == 0:
    #     plt.cla()  # 清坐标轴
    #     plt.scatter(x.data.numpy(), y.data.numpy())
    #     plt.plot(x.data.numpy(), predication.data.numpy(), 'ro', lw=5)  # 画预测曲线，用红色o作为标记
    #     plt.text(0.5, 0, 'Loss = %.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
    #     plt.pause(0.1)
