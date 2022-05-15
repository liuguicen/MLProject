# 1、一键导入常用的库，不用重复写代码
# 2、进行常见的设置，比如plt支持中文

# 然后直接使用相关类或者文件即可

# 用法 from ml_base.common_lib_import import *

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torchvision.models import MobileNetV3
from torch.utils.data import Dataset

# python
import os
from os import path
import random

# 三方
import pandas
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2

# 自己的
import FileUtil
import image_util
import video_util
import CommonConstantParam

import common_dataset

# 设置plt支持中文
plt.rcParams['font.sans-serif'] = ['simhei']  # simhei 这个字体需要先手动下载放到plt的目录下，参考网上方法
plt.rcParams['axes.unicode_minus'] = False
if __name__ == '__main__':
    video_util
    os
    FileUtil
    image_util
    torch
    Image
    cv2
    common_dataset
    random.random()
    path.join('a', 'b')
    np.ndarray
    # torch
    torch
    nn
    optim
    torchvision
    transforms
    datasets, models, transforms
    MobileNetV3
    Dataset
    random
    pandas
    CommonConstantParam