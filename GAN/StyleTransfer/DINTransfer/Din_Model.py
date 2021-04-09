import torch.nn as nn
import torch.nn.functional as F
import torchvision

import Din_Config
from ml_base.CommonModels.CommonModels import Vgg


class StyleConv(nn.Sequential):
    '''风格网络中所使用的卷积层，并且移动化， pad使用镜像pad，然后relu使用relu6
    关于使用镜像padding：
    风格模型基本上都是用的镜像padding，adain里面原话：
    We use reflection padding in both f（编码器） and g（解码器） to avoid border artifacts.
    有论文解释了，好像是AdaIN
    解码器肯定是镜像，对于编码器
    看到的在LinearStyleTransfer  WCT2 编码器也是用的镜像，其它VGG 或者 还没看到

    关于是否添加normal：
    这个存疑，有的论文用了，有的没用比如adain
    以及使用那种normal

    激活层：仿照mobilenet 使用relu6
    '''

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, groups=1, norm_layer=Din_Config.normal,
                 active_layer=Din_Config.active_layer):
        padding = (kernel_size - 1) // 2
        self.sequential = nn.Sequential()
        if padding != 0:
            self.sequential.add_module('pad', nn.ReflectionPad2d((padding, padding, padding, padding)))

        self.sequential.add_module('conv',
                                   nn.Conv2d(in_channel, out_channel, kernel_size, stride, groups=groups, bias=False))

        if norm_layer is not None:
            self.sequential.add_module('normal', norm_layer(out_channel))

        if active_layer is not None:
            self.sequential.add_module('relu', active_layer)

    def forward(self, x):
        return self.sequential(x)


class Bottleneck(nn.Module):
    ''' 参考pytorch resnet 官方代码'''

    # type:torchvision.models.ResNet

    def __init__(self, input_channel, middle_channel):
        '''
        :param middle_channel: 没有特别处理时就是，缩小之后的通道数量
        dilation 应该就是
        '''
        super(Bottleneck, self).__init__()
        self.conv1 = StyleConv(input_channel, middle_channel, kernel_size=1)
        self.conv2 = StyleConv(middle_channel, middle_channel, kernel_size=3)
        self.conv3 = StyleConv(middle_channel, input_channel, kernel_size=1, active_layer=None)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += identity
        out = self.relu(out)

        return out


class MobileBased_Encoder(nn.Module):
    '''
        文中所述的MobileNet-based Network， 显然的基于MobileNet，不是直接用的MobileNet
    '''

    def __init__(self):
        nn.Module.__init__(self)
        # 一个标准卷积
        self.conv = StyleConv(3, 16, kernel_size=9)

        # 两个深度可分离
        self.deep1 = StyleConv(16, 32, kernel_size=3, stride=2, groups=16)
        self.point1 = StyleConv(32, 32, kernel_size=1)

        self.deep2 = StyleConv(32, 64, kernel_size=3, stride=2, groups=32)
        # 这里存疑，深度可分离卷积的kernel_size 应该是1，但材料中给出的是3
        self.point2 = StyleConv(64, 64, kernel_size=Din_Config.pw2_kernal_size)

        # 两个resblock
        self.resLayer1 = Bottleneck(64, 16)
        self.resLayer2 = Bottleneck(64, 16)

    def forward(self, x):
        x = self.conv(x)
        x2 = self.point1(self.deep1(x))
        x1 = self.point2(self.deep2(x2))
        x1 = self.resLayer1(x1)
        x1 = self.resLayer2(x1)
        # 有多个DIN层，需要不同尺度的feature
        return x1, x2


class MobileNet_Based_Decoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        # 两个resblock
        self.feature1 = Bottleneck(16, 64)
        self.feature2 = Bottleneck(64, 16)

        # 两个深度可分离
        self.deep1 = StyleConv(64, 32, kernel_size=3, groups=32)  # 深度卷积
        self.point1 = StyleConv(32, 32, kernel_size=1)  # 逐点卷积

        self.deep2 = StyleConv(32, 16, kernel_size=3, stride=2, groups=16)
        self.point2 = StyleConv(16, 16, kernel_size=1)

        # 一个标准卷积
        # 最后一层不激活，大多数st模型都是这样做的
        self.feature5 = StyleConv(16, 3, kernel_size=9,
                                  active_layer=nn.functional.tanh if Din_Config.useTanh else None)

    def forward(self, x1, x2):
        x = self.feature1(x1)
        x = self.feature2(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.point1(self.deep1(x))

        x = x + x2
        x = F.interpolate(x, scale_factor=2)
        x = self.point2(self.deep2(x))

        x = self.feature5(x)
        return x


class Weight_Bias_Net(nn.Module):
    def __init__(self, input_channel):
        nn.Module.__init__(self)
        # 文中提到，默认din 过滤器size设置为1，为了减少计算消耗，但是附录里面卷积核大小是3
        self.layer1 = StyleConv(input_channel, 128, kernel_size=Din_Config.dinLayer_filterSize, stride=2, groups=128)

        self.layer2 = StyleConv(128, 64, kernel_size=Din_Config.dinLayer_filterSize, stride=2, groups=64)

        self.layer3 = StyleConv(64, 64, kernel_size=Din_Config.dinLayer_filterSize, stride=2, groups=64)

    def forward(self, x, output_size: int):
        x = self.layer1(x)
        x = self.layer2(x)

        x = Din_Config.weight_bias_pool_layer(x, Din_Config.get_adapool1_output_size(output_size))
        x = self.layer3(x)
        # 输出的通道数量 = 后面要处理的通道数量 = 卷积核个数（每个通道的参数给到一个卷积核？）
        # type: nn.AdaptiveMaxPool2d()
        x = Din_Config.weight_bias_pool_layer(x, output_size)
        return x


class DIN_layer(nn.Module):
    def __init__(self, content_channel: int, style_channel: int):
        nn.Module.__init__(self)
        self.IN = nn.InstanceNorm2d(content_channel)
        self.weightN = Weight_Bias_Net(style_channel)  # 输出size = 输入size
        self.biasN = Weight_Bias_Net(style_channel)

    def forward(self, x, din_out_size: int, s_feature=None, para_w=None, para_b=None):
        if s_feature == None:  # 推理过程，风格为空，使用参数
            para_w = self.weightN(s_feature, din_out_size)
            para_b = self.biasN(s_feature, din_out_size)
        # para_w 的尺寸 = batch * 64 * din_out_size
        # para_b 的尺寸 = batch * 64 * din_out_size = para_w
        # x      的尺寸 = batch * 64 * din_out_size
        x = self.IN(x)
        x = x * para_w + para_b
        return x


class DINModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = MobileBased_Encoder()

        self.vgg = Vgg()
        self.dinLayer1 = DIN_layer(64, Vgg.feature_channel)
        self.dinLayer2 = DIN_layer(64, Vgg.feature_channel)

        self.decoder = MobileNet_Based_Decoder()

    def forward(self, content, style):
        cFeature = self.encoder(content)
        styleFeature = self.vgg(style)

        dinFeature1 = self.dinLayer1(cFeature[0], cFeature[0].size()[2:3] / 4, styleFeature)
        dinFeature2 = self.dinLayer2(cFeature[1], cFeature[1].size()[2:3] / 4, styleFeature)

        out = self.decoder(dinFeature1, dinFeature2)
        return out
