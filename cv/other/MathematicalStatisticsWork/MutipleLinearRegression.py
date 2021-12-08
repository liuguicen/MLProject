import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data_process

# 定义最高次项系数
n = 1
torch.set_default_tensor_type(torch.DoubleTensor)

# 定义模型
class LinerRegression(nn.Module):
    def __init__(self):
        super(LinerRegression, self).__init__()
        self.poly = nn.Linear(n, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        out = self.poly(x)
        return out


# 定义函数输出形式
def func_format(weight, bias, n):
    func = ''
    for i in range(n, 0, -1):
        func += ' {:.2f} * x^{} +'.format(weight[i - 1], i)
    return 'y =' + func + ' {:.2f}'.format(bias[0])


model = LinerRegression()

# 损失函数和优化器
criterion = nn.MSELoss()

epoch = 0
loss_list = []
while epoch < 45:
    # 获得数据
    batch_x, batch_y = data_process.get_next_batch(20)
    # 向前传播
    if batch_y.size()[0] < 2:
        continue
    output = model(batch_x)
    print("模型结果", Variable(output).numpy().flatten() * 65163)
    print("实际结果", Variable(batch_y).numpy().flatten() * 65163)
    loss = criterion(output, batch_y)
    # 重置梯度
    model.optimizer.zero_grad()
    # 后向传播
    loss.backward()
    # 更新参数
    model.optimizer.step()
    epoch += 1
    print_loss = loss.item()
    loss_list.append(print_loss)
    print("epoch = ", epoch, "loss = ", print_loss)
# if print_loss < 1e-3:
#     break

print("the number of epoches :", epoch)
predict_weight = model.poly.weight.data.cpu().numpy().flatten()
print(predict_weight)
predict_bias = model.poly.bias.data.cpu().numpy().flatten()
print(predict_bias)

plt.plot(loss_list)
plt.show()
# visulize()
