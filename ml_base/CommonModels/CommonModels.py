import torch
import torchvision
from torch import nn as nn

import MlUtil


class MyVgg(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        vgg = torchvision.models.vgg19(pretrained=True).features[:21]
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input, needMltiLayer: bool = False):
        h1 = self.slice1(input)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if not needMltiLayer:
            return h4  # 刚好第4段第一层卷积后的激活，即relu4_1, 通道数 feature_channel = 512
        else:
            return h1, h2, h3, h4


class MyVgg_Inference(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        vgg = torchvision.models.vgg19(pretrained=True).features[:21]
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input):
        h1 = self.slice1(input)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h4  # 刚好第4段第一层卷积后的激活，即relu4_1, 通道数 feature_channel = 512


class MyMobileNet(nn.Module):
    outChannel = 384

    def __init__(self):
        nn.Module.__init__(self)
        # type:torchvision.models.MobileNetV2
        moblieNet = torchvision.models.mobilenet_v2(pretrained=False)
        moblieNet.load_state_dict(torch.load(path.join(common_dataset.dataset_dir, r'预训练模型\mobilenet_v2-b0353104.pth'))
        self.slice1 = moblieNet.features[:1]
        self.slice2 = moblieNet.features[1:2]
        self.slice3 = moblieNet.features[2:4]
        self.slice4 = moblieNet.features[4:7]
        self.slice5 = moblieNet.features[7:11]
        # 取到1 / 8 原图大小的特征图，这个图从mobileNet的反向resblock的中间取，详见mobile结构图
        self.slice5.add_module('layer in inverseRes', moblieNet.features[11].conv[:2])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, return_last=True):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        if return_last:
            return h5
        else:
            return h1, h2, h3, h4, h5


def use():
    # 保持导入的，不删
    pass


if __name__ == "__main__":
    net = MyMobileNet()
    # net = MyVgg()
    net(MlUtil.readTestPicBatch(r'D:\MLProject\ml_base\111.png'))
