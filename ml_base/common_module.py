import torch.nn as nn
import torch


class ConvLayer(nn.Module):
    def __init__(self, in_channle, out_channel, k, p, s=1, normal=nn.BatchNorm2d, active=nn.ReLU):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels=in_channle, out_channels=out_channel, kernel_size=k, padding=p, stride=s)
        self.normal = normal(out_channel)
        self.active = active

    def forward(self, x):
        x = self.conv(x)
        x = self.normal(x)
        x = self.active(x)
