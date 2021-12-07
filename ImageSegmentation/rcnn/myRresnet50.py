import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 论文里面的3个卷积层的残差块，因为不同卷积层大小不同，形状类似瓶颈
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cur_out_channel, outchannel, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # 第一个卷积层
        self.conv1 = conv1x1(cur_out_channel, outchannel)
        self.bn1 = norm_layer(outchannel)

        # 第二个卷积层
        self.conv2 = conv3x3(outchannel, outchannel, stride)
        self.bn2 = norm_layer(outchannel)

        # 第三个卷积层
        self.conv3 = conv1x1(outchannel, outchannel * self.expansion)
        self.bn3 = norm_layer(outchannel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 进个第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 经过第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 经过第三个卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # 输入和卷积输出进行合并
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # block 就是残差块类对象
    def __init__(self, layers, num_classes=1000):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # 规范层

        self.cur_out_channel = 64
        '''resnet构造的当前的卷积层的输出通道的数量，从64开始'''

        # 第一个7*7卷积层
        self.conv1 = nn.Conv2d(3, self.cur_out_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.cur_out_channel)  # 规范化
        self.relu = nn.ReLU(inplace=True)  #
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #

        # 这下面的堆叠层实际上就是根据尺寸不同而分开的
        self.layer1 = self._make_layer(64, layers[0])  # 第一个残差堆叠层
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # 初始化，自动获取所有的实例的属性，判断是否是需要初始化的层然后加以初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 创建残差的堆叠层
    # block残差块类对象，blocks残差块的数量
    # layer_channel 为通道数量
    def _make_layer(self, layer_channel, blocks, stride=1):

        norm_layer = self._norm_layer
        downsample = None
        # 这个是调整短连接的尺寸的，因为堆叠层内部尺寸是相同的，所以只需要开始的地方调整一次就行了

        if stride != 1 or self.cur_out_channel != layer_channel * Bottleneck.expansion:  # 新的层通道数量和当前通道数不同
            downsample = nn.Sequential(
                conv1x1(self.cur_out_channel, layer_channel * Bottleneck.expansion, stride),
                norm_layer(layer_channel * Bottleneck.expansion),
            )

        layers = []
        # 创建一个残差块并加入列表
        # 第一个残差块，可能需要增加短连接的尺寸
        # type: Bottleneck.__init__()
        # type: BasicBlock.__init__()
        layers.append(
            Bottleneck(self.cur_out_channel, layer_channel, stride=stride, downsample=downsample, norm_layer=norm_layer))
        self.cur_out_channel = layer_channel * Bottleneck.expansion
        # 剩余的若干个残差块
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.cur_out_channel, layer_channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(layers):
    model = ResNet(layers)
    state_dict = torch.load(r'E:\data_big_size\pretrain_weight\resnet50-19c8e357.pth')
    model.load_state_dict(state_dict)
    return model


def resnet50():
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet([3, 4, 6, 3])
