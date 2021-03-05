import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道， '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层，输入1350个特征，输出10个特征
        self.fc1 = nn.Linear(2, 4)  # 这里的1350是如何计算的呢？这就要看后面的forward函数

    # 正向传播
    def forward(self, x):
        print(x.size())  # 结果：[1, 1, 32, 32]
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)  # 根据卷积的尺寸计算公式，计算结果是30，具体计算公式后面第二章第四节 卷积神经网络 有详细介绍。
        x = F.relu(x)
        print(x.size())  # 结果：[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
        x = F.relu(x)
        print(x.size())  # 结果：[1, 6, 15, 15]
        # reshape，‘-1’表示自适应
        # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        print(x.size())  # 这里就是fc1层的的输入1350
        x = self.fc1(x)
        return x


net = Net()
print(net)
print('paras = \n', net.parameters())
for paras in net.named_parameters():
    print(paras)