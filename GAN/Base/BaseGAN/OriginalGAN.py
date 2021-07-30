import time
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
import common_dataset
import math
import run_record
from ml_base import *
from ml_base.run_record import RunRecord
import os
from os import path

from common_dataset import Minist
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms


class Config:
    checkPointCount = 20
    batch_size = 256
    epoch = 20
    '''
    输入生成器的尺寸
    '''
    # input_w = 10
    input_w = 10
    # 生成器每一层尺寸
    # layers = [4 * 4, 7 * 7, 14 * 14, 28 * 28]
    layers = [input_w * input_w, 128, 256, 512, 1024, 784]
    '''
    判别器每一层的深度
    '''
    depth = [8, 16, 32]



from CommonModels import NormalConvLayer


class Conv3x3(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        nn.Module.__init__(self)
        super().add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1))
        super().add_module("bn", nn.BatchNorm2d(out_channel))
        super().add_module('relu', nn.LeakyReLU())


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential()
        for i, w in enumerate(Config.layers):
            if i < len(Config.layers) - 1:
                self.seq.add_module('linear %d - %d' % (Config.layers[i], Config.layers[i + 1]),
                                    nn.Linear(Config.layers[i], Config.layers[i + 1]))
                if i != 0:
                    self.seq.add_module("bn %d" % (i + 1), nn.BatchNorm1d(Config.layers[i + 1], 0.8))
                if i < len(Config.layers) - 2:
                    self.seq.add_module('relu%d' % i, nn.LeakyReLU())
                else:
                    self.seq.add_module('last tanh', nn.Tanh())
                    # 相对于singmod值区间是0-1，tanh和sigmod是线性关系，但是它直接将区间变到-1 - 1

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size()[0], 1, 28, 28)
        # x = x * 2 - 1
        return x


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#
#         return validity


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_seq = nn.Sequential()
        for i, dep in enumerate(Config.depth):
            if i == 0:  # 第一层，单通道办成dep通道
                self.conv_seq.add_module('conv layer %d - %d' % (1, dep),
                                         Conv3x3(1, dep))
            else:
                self.conv_seq.add_module('conv layer %d - %d' % (dep // 2, dep),
                                         Conv3x3(dep // 2, dep))
            # if i < len(Config.depth) - 1:
            #     self.conv_seq.add_module('pool %d' % i, nn.MaxPool2d(2))
        # 输入尺寸是28 * 28，
        self.linear = nn.Linear(4 * 4 * Config.depth[-1], 4 * 4 * Config.depth[-1] // 4)
        self.out_lin = nn.Linear(4 * 4 * Config.depth[-1] // 4, 1)

    def forward(self, x: torch.Tensor):
        '''
        黑白数字，单通道
        '''
        x = self.conv_seq(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.out_lin(x)
        x = torch.sigmoid(x)
        return x


import ml_util

D = Discriminator()
D.to(ml_util.gpu)
# print(D)
G = Generator()
G.to(ml_util.gpu)
# print(G)


dOpt = Adam(D.parameters(), lr=0.0002)
gOpt = Adam(G.parameters(), lr=0.0002)
lossFunction = nn.BCELoss()


def train():
    minist = Minist()
    dataLoader = DataLoader(minist, batch_size=Config.batch_size, shuffle=True)
    G_label = torch.ones([Config.batch_size]).unsqueeze(1).to(ml_util.gpu)
    D_label = torch.cat([torch.ones([Config.batch_size]), torch.zeros([Config.batch_size])], 0).unsqueeze(1).to(
        ml_util.gpu)

    test_z = torch.randn([8, Config.input_w * Config.input_w]).to(ml_util.gpu)
    lossGList = []
    lossDList = []
    runRecord = RunRecord.readFromDisk()
    if runRecord is None:
        runRecord = RunRecord()

    d_step = 0
    lossG = None
    for epoch in range(Config.epoch):
        print(f'start {epoch} epoch')
        for iter, item in enumerate(dataLoader):
            # 训练判别器k步
            d_step += 1
            image = item[0].to(ml_util.gpu)
            if image.size()[0] != Config.batch_size:  # 最后一批
                continue
            p_data = D(image)

            z = torch.randn([Config.batch_size, Config.input_w * Config.input_w]).to(ml_util.gpu)
            g_image = G(z)  # type:torch.Tensor
            g_image.detach()
            p_g = D(g_image)
            D_out = torch.cat([p_data, p_g], 0)
            lossD = lossFunction(D_out, D_label)  # type:torch.Tensor
            dOpt.zero_grad()
            D.zero_grad()
            lossD.backward()
            dOpt.step()

            if d_step < 1:
                continue
            d_step = 0
            print('优化D完成')
            print('p_data = ', D(image).mean().item(), 'p_g = ', D(g_image).mean().item())
            print('loss G = ', lossFunction(p_g, G_label).item(), 'loss D = ', lossD.item())

            # 训练生成器一步
            for i in range(1):  # 分类器能力太强，多跑几步生成器
                z = torch.randn([Config.batch_size, Config.input_w * Config.input_w]).to(ml_util.gpu)
                g_image = G(z)
                p_g = D(g_image)
                lossG = lossFunction(p_g, G_label)  # type:torch.Tensor
                gOpt.zero_grad()
                G.zero_grad()
                lossG.backward()
                gOpt.step()

            print('优化G完成')
            print('p_data = ', D(image).mean().item(), 'p_g = ', D(g_image).mean().item())
            print('loss G = ', lossG.item(), 'loss D = ', lossD.item())
            lossGList.append(lossG.item())
            lossDList.append(lossD.item())

            if iter % Config.checkPointCount == 0:
                # torchvision.utils.save_image(image[0:8], f'{runRecord.tes_res_dir}/{epoch}_{iter}_real.jpg')

                testOut = G(test_z)
                torchvision.utils.save_image(testOut, f'{runRecord.tes_res_dir}/{epoch}_{iter}_generate.jpg')
                runRecord.saveRunRecord(epoch, iter)


if __name__ == "__main__":
    train()
