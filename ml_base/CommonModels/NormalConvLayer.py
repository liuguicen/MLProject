import torch
from torch import nn


class NormalConv3x3(nn.Sequential):

    def __init__(self, in_channel, out_channel):
        nn.Module.__init__(self)
        super().add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        super().add_module('relu', nn.ReLU())
