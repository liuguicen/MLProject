import time

import torch


class Const:
    '''
    常用字符串等，避免写错
    '''
    total = 'total'
    default = 'defalt'


class Timer:
    '''
    recordkey 用于需要多个记录时
    '''
    startT = {Const.default: time.time()}

    @classmethod
    def record(cls, recordkey=Const.default):
        cls.startT[recordkey] = time.time()

    @classmethod
    def print_and_record(cls, msg='', recordKey=Const.default):
        cls.print(msg, recordKey)
        cls.record(recordKey)

    @classmethod
    def print(cls, msg='', recordKey=Const.default):
        print(msg, time.time() - cls.startT[recordKey])


def viz(module, input):
    x = input[0][0]  # 现在的x是一个样本
    showSample(x,  4, module)


def showSample(x, feature_number, title):
    x = x.detach().cpu()
    # 最多显示4张图
    min_num = np.minimum(feature_number, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i + 1)
        plt.imshow(x[i])  # x[i] 就是一个特征图
    plt.title(title)
    plt.show()


import torchvision
import numpy as np
from matplotlib import pyplot as plt


def printMiddleFeature(x):
    x = x[0].detach()  # 取一个样本
    showSample(x, 5, '')

def registerMiddleFeaturePrinter(model):
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)


def printAllMiddleFeature(model, *img):
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)

    with torch.no_grad():
        model(img[0], img[1])


if __name__ == '__main__':
    Timer.print_and_record()
