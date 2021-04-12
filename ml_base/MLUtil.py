import time

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
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i])
    plt.show()


import torchvision
import numpy as np
from matplotlib import pyplot as plt

def printMiddleFeature(x):
    x = x.detach() # 核心代码
    x = x.transpose(1,0).cpu()
    inp = torchvision.utils.make_grid(x)
    """Imshow for Tensor."""

    inp = inp.detach().numpy().transpose((1, 2, 0))

    mean = np.array([0.5, 0.5, 0.5])

    std = np.array([0.5, 0.5, 0.5])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    plt.pause(0.001)  # pause a bit so that plots are updated




if __name__ == '__main__':
    Timer.print_and_record()
