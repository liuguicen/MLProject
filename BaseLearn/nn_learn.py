import torch
import torch.nn as nn
import torch.nn.functional as F


# pytorch 中定义神经网络都需要Module，它是基类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 13, 5)  # 二维的卷积，这两个属性应该是自带的，
        # 还可以使用add_module将添加到额外的层到网络中。
        # 如
        # self.add_module("maxpool1", nn.MaxPool2d(2, 2))
        # self.add_module("covn3", nn.Conv2d(64, 128, 3))
        # self.add_module("conv4", nn.Conv2d(128, 128, 3))
        # 在正向传播的过程中可以使用添加时的name来访问改layer。
        # 这就需要使用到torch.nn.ModuleList和torch.nn.Sequential。
        #
        # # 使用ModuleList和Sequential可以方便添加子网络就是多个层到网络中，但是这两者还是有所不同的。
        # ModuleList可以将一个Module的List加入到网络中，自由度较高，但是需要手动的遍历ModuleList进行forward。
        # Sequential按照顺序将将Module加入到网络中，也可以处理字典。 相比于ModuleList不需要自己实现forward
        self.conv2 = nn.Conv2d(13, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # relu为激活函数，相当于吴恩达课程中的由z得到a
        # 池化，就是采样，减少“图片”尺寸大小，采用规则应该就是2*2的区域变成1
        # max_pool2d表示取最大值，对应的还有avg_pool等
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 第二层卷积，激活，采样
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 将张量变成一维向量的形式
        x = x.view(-1, self.num_flat_features(x))
        # 这一层的输出，乘上参数矩阵，得到下一层的输入，再经过relu函数，变成下一层的输出
        # 参数矩阵行数等于输入向量的列数，参数矩阵的列数等于输出向量的列数，最后得到输出的一个行向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print('网络结构：\n', net)
print('参数结构:')
for k, v in net.named_parameters():
    print(k, v.size())
# params = list(net.parameters())
# print('parameter is ', params)
# print(params[0].size())  # conv1's .weight
input = torch.randn(1, 1, 32, 32)  # 参数为（样本数，通道数，尺寸）
out = net(input)
target = torch.randn(10).view(1, -1)
loss = nn.MSELoss()(out, target)
print('损失值为\n', loss)
# 关于grad_fn，grad_fn 可以把一个计算公式看成一棵树结构，只有叶子节点可以计算导数，grad_fn.next_functions就是某个节点的分支节点
# print(' 梯度对象为\n', loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# 应该是多次计算的grad会累加，比如两个批次的训练数据，所以训练下一次数据之前，要先置0
# 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。
# 其实这里还可以补充的一点是，如果不是每一个batch就清除掉原有的梯度，而是比如说两个batch再清除掉梯度，这是一种变相提高batch_size的方法，对于计算机硬件不行，但是batch_size可能需要设高的领域比较适合，比如目标检测模型的训练。
net.zero_grad()  # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
# bias偏置向量
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# 更新参数
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import nn_learn_data
