import torch.nn as nn
import torch.nn.functional

import CommonDataSet
from transformer_net import ConvLayer

start_channel = 8


class Generator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):  # x是一维随机向量


import CommonModule


class Discriminator(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = CommonModule.ConvLayer(3, start_channel, k=3, p=1, s=1)
        self.conv2 = CommonModule.ConvLayer(start_channel, start_channel * 2, k=3, p=1)
        self.conv3 = CommonModule.ConvLayer(start_channel * 2, start_channel * 4, k=3, p=1)
        self.conv4 = CommonModule.ConvLayer(start_channel * 4, start_channel * 8, k=3, p=1)
        self.conv5 = CommonModule.ConvLayer(start_channel * 8, start_channel * 16, k=3, p=1)
        self.conv6 = CommonModule.ConvLayer(start_channel * 16, start_channel * 32, k=3, p=1)
        self.conv7 = CommonModule.ConvLayer(start_channel * 32, start_channel * 64, k=3, p=1)
        self.linear1 = nn.Linear(start_channel * 64, start_channel * 64)
        self.linear2 = nn.Linear(start_channel * 64, 1)

    def forward(self, x):
        # 3*64*64
        x = self.conv1(x)
        x = nn.MaxPool2d(x, 2)
        # 8*64*64
        x = self.conv2(x)
        x = nn.MaxPool2d(x, 2)
        # 16*32*32
        x = self.conv3(x)
        x = nn.MaxPool2d(x, 2)
        # 32*16*16
        x = self.conv4(x)
        x = nn.MaxPool2d(x, 2)
        # 64*8*8
        x = self.conv5(x)
        x = nn.MaxPool2d(x, 2)
        # 128*4*4
        x = self.conv6(x)
        x = nn.MaxPool2d(x, 2)
        # 256*2*2
        x = self.conv7(x)
        # 512*1*1
        x = self.linear1(x)
        x = nn.ReLU(x)
        # 512*1*1
        x = self.linear2(x)
        # 1 * 1 * 1
        return torch.nn.functional.sigmoid(x)


import torch.utils.data

def loss(y, label):



def test():
    dataset = CommonDataSet.Minist()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    D = Discriminator()
