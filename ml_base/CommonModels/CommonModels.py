import torchvision
from torch import nn as nn


class Vgg(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        vgg = torchvision.models.vgg19(pretrained=True).features[:21]
        vgg.to('cpu')
        vgg.cpu()
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
        return h4 # 这里只使用第四段的，有些情况下3段的都要使用