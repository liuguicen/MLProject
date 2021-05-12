import time

import torch

import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


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
    showOrSaveSample(x, 8, module)


def showOrSaveSample(x, feature_number, title, savePath=None):
    x = x.detach().cpu()
    # 最多显示4张图
    min_num = np.minimum(feature_number, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, feature_number, i + 1)
        plt.imshow(x[i])  # x[i] 就是一个特征图
    plt.title(title)
    if savePath is None:
        plt.show()
    else:
        plt.savefig(savePath)


def printMiddleFeature(x):
    x = x[0].detach()  # 取一个样本
    showOrSaveSample(x, 5, '')


def saveMiddleFeature(x, feature_number, title, path):
    x = x[0].detach()  # 取一个样本
    showOrSaveSample(x, feature_number, title, path)


def registerMiddleFeaturePrinter(model):
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)


def printAllMiddleFeature(model, *img, type=torch.nn.Conv2d):
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, type):
            m.register_forward_pre_hook(viz)

    with torch.no_grad():
        model(*img)


def printDivide():
    plt.plot([5, 5, 5, 5])
    plt.show()


def readTestPicBatch(path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([transforms.ToTensor(),
                                normalize])
    i = Image.open(path)
    return trans(i).unsqueeze(0).to(gpu)


gpu = 'cuda:0'
if __name__ == '__main__':
    Timer.print_and_record()


def use():
    # 占位
    pass
