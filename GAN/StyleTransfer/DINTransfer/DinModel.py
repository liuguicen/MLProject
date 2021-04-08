import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet
from torchvision.models.resnet import Bottleneck

from ml_base.CommonModels.CommonModels import Vgg


class DIN_layer(nn.Module):
    def __init__(self, input_channel: int):
        nn.Module.__init__(self)
        self.IN = nn.InstanceNorm2d(input_channel)
        self.weightN = Weight_Bias_Net(input_channel)  # 输出size= 输入size
        self.biasN = Weight_Bias_Net(input_channel)

    def forward(self, x, din_out_size: int, s_feature=None, para_w=None, para_b=None):
        if s_feature == None:  # 推理过程，风格为空，使用参数
            para_w = self.weightN(s_feature, din_out_size)
            para_b = self.biasN(s_feature, din_out_size)
        x = self.IN(x)
        x = x * para_w + para_b
        return x


class Weight_Bias_Net(nn.Module):
    def __init__(self, input_channel):
        nn.Module.__init__(self)
        self.layer1 = StyleLayer(input_channel, 128, kernel_size=3, stride=2, groups=128)

        self.layer2 = StyleLayer(128, 64, kernel_size=3, stride=2, groups=64)

        self.layer3 = StyleLayer(64, 64, kernel_size=3, stride=2, groups=64)

    def forward(self, x, output_size: int):
        x = self.layer1(x)
        x = self.layer2(x)
        # 因为输入的大小不固定，输出是固定的，所以采用自适应池化AdaptiveXxxPool，pytorch包装了，自己换算也可以的
        # 文中没说用那种池化，网上说最大池化保留纹理信息，应该是这个
        # 根据卷积的输出size，反向计算输入size, 让其经过layer3之后是outputSize的两倍，先乘以2，再算出卷积前的大小
        # 卷积尺寸公式：size_out = (size_in + 2 * pad - k) / s + 1
        adapool1_output_size = (output_size * 2 - 1) * 2 + 1
        x = F.adaptive_max_pool2d(x, adapool1_output_size)
        x = self.layer3(x)
        # 输出的通道数量 = 后面要处理的通道数量 = 卷积核个数（每个通道的参数给到一个卷积核？）
        # type: nn.AdaptiveMaxPool2d()
        x = F.adaptive_max_pool2d(x, output_size)
        return x


class MobileBased_Encoder(nn.Module):
    '''
        文中所述的MobileNet-based Network， 显然的基于MobileNet，不是直接用的MobileNet
    '''

    def __init__(self):
        nn.Module.__init__(self)
        # 一个标准卷积
        self.feature1 = mobilenet.ConvBNReLU(3, 16, kernel_size=9)

        # 两个深度可分离
        self.deep1 = mobilenet.ConvBNReLU(16, 32, kernel_size=3, stride=2, groups=16)
        self.feature2 = mobilenet.ConvBNReLU(32, 32, kernel_size=1)

        self.deep2 = mobilenet.ConvBNReLU(32, 64, kernel_size=3, stride=2, groups=32)
        self.feature3 = mobilenet.ConvBNReLU(64, 64, kernel_size=3)

        # 两个resblock
        self.feature4 = Bottleneck(64, 16)
        self.feature5 = Bottleneck(64, 16)

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(self.deep1(x))
        x = self.feature3(self.deep2(x))
        x = self.feature4(x)
        x = self.feature5(x)
        return x


class StyleLayer(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d):
        padding = (kernel_size - 1) // 2
        super(StyleLayer, self).__init__(
            nn.ReflectionPad2d((padding, padding, padding, padding)),  # 模型基本上都是用的这个加padding
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, groups=groups, bias=False),
            # norm_layer(out_planes), # 是否添加normal， 这个存疑，有的用的，有的没用adain
            nn.ReLU6(inplace=True)
        )


class MobileNet_Based_Decoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # 两个resblock
        self.feature1 = Bottleneck(16, 64, base_width=16)
        self.feature2 = Bottleneck(64, 16)

        # 两个深度可分离
        self.dw2 = StyleLayer(64, 32, kernel_size=3, groups=32)  # 深度卷积
        self.pw1 = StyleLayer(32, 32, kernel_size=1)  # 逐点卷积

        self.dw2 = StyleLayer(32, 16, kernel_size=3, stride=2, groups=16)
        self.pw2 = StyleLayer(16, 16, kernel_size=1)

        # 一个标准卷积
        self.feature5 = nn.Sequential(
            nn.ReflectionPad2d((4, 4, 4, 4)),
            nn.Conv2d(16, 3, kernel_size=9))  # 最后一层不激活，大多数st模型都是这样做的

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.dw1(x)
        x = self.pw1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.dw2(x)
        x = self.pw2(x)

        x = self.feature5(x)
        return x


def compute_loss(vgg, out, content, style):
    styleFeature = vgg(style)
    contentFeature = vgg(content)
    outFeature = vgg(out)

    return None


import torch


def forward(content, isTrain: bool, style=None, para_w=None, para_b=None):
    encoder = MobileBased_Encoder()
    decoder = MobileNet_Based_Decoder()
    dinLayer = DIN_layer(64)
    optimizer = torch.optim.Adam(encoder.parameters())
    vgg = Vgg()
    if isTrain:
        optimizer.zero_grad()
        content = encoder(content)
        styleFeature = vgg(style)
        feature = dinLayer(content, content.size()[2:3] / 4, styleFeature)
        out = decoder(feature)
        loss = compute_loss(vgg, out, content, style)
        loss.backward()
        optimizer.step()
