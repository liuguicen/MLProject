from torch import nn
from torch.nn import Module

'''
写这个的经验：
1、编写神经网络的时候，会涉及到很多数字，比如长宽，通道，核size，stride等等
多则乱，并且还得配合网络的各种结构，这上面很容易出错，自己第一次写这里不知道错了多少个地方
所以一定一定要注意把数字写对！
具体做法，应该是先画好神经网络的结构图，确保结构图正确，然后每次写代码的时候和结构图对比确认，完全一致才行
出错了，从结构图检查应该更好

2、还有一个神经网络涉及到的组件很多，不要写漏了，比如少写一个bn，少写一个relu什么的

这是NN编程的主要难点之一吧
'''


#  一个残差块，两层的

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        nn.Module.__init__(self)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:  # 短连接的扩增，宽度不同（stride != 1)，或者通道不同
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        x = self.convLayer1.forward(x)
        x = self.convLayer2.forward(x)

        # match the size of shortcut connect
        # 此处可以改进，采用先relu激活，后相加的方法
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        return x

class MyResNet(Module):

    # 创建残差块的组
    def makeRbGroup(self, rbNumber, in_channels, out_channels, stride, kernel_size=3, padding=1):
        rbList = []
        for i in range(rbNumber):
            # the first resBlock's stride != 1 for scale size
            rbList.append(ResBlock(in_channels if i is 0 else out_channels, out_channels, kernel_size, stride if i == 0 else 1, padding))
        return nn.Sequential(*rbList)

    def __init__(self):
        Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activ1 = nn.ReLU(inplace=True)
        # cur size is 64*112*112
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 此时尺寸为 64 * 56 * 56
        self.groupSize = [2, 2, 2, 2]
        self.group1 = self.makeRbGroup(2, in_channels=64, out_channels=64, stride=1)
        # 此时尺寸为 128* 28 * 28
        self.group2 = self.makeRbGroup(2, in_channels=64, out_channels=128, stride=2)
        # 此时尺寸为 256 * 14 * 14
        self.group3 = self.makeRbGroup(2, in_channels=128, out_channels=256, stride=2)
        # 此时尺寸为 512 * 7 * 7
        self.group4 = self.makeRbGroup(2, in_channels=256, out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3),
            nn.LogSoftmax(dim=1)
        )

        # 初始化，自动获取所有的实例的属性，判断是否是需要初始化的层然后加以初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activ1(x)
        x = self.pool1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        return x
