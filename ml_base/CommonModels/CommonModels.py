import torch
import torchvision
from torch import nn as nn


class Vgg(nn.Module):

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


def use():
    # 保持导入的，不删
    pass