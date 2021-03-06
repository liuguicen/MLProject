# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:07:23 2019
@author: AugustMe
"""
import numpy as np
import os
import gzip

import torch.utils.data
from torchvision import transforms
dataset_dir = 'E:\重要_dataset_model_数据集'
from os import path

wikiartPath = path.join(dataset_dir, r'\wikiart\train')
animatePath = path.join(dataset_dir, r'l\动画漫画\动画漫画')
cocoPath = path.join(dataset_dir, r'l\COCO\train2014')

ministTransforms = transforms.Normalize((0.1307,), (0.3081,))

class Minist(torch.utils.data.Dataset):
    def load_data(self, dir, isTest: bool = False):
        self.isTest = isTest
        files = [
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
        ]

        paths = []
        for fname in files:
            paths.append(os.path.join(dir, fname))

        with gzip.open(paths[0], 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[1], 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

        with gzip.open(paths[2], 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[3], 'rb') as imgpath:
            x_test = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

        return x_train, y_train, x_test, y_test

    # 定义加载数据的函数，data_folder为保存gz数据的文件夹，该文件夹下有4个文件
    # 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
    # 't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'

    def __init__(self, isTest=False):
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_data(
            os.path.join(dataset_dir, 'MNIST'), isTest)  # type:torch.tensor

        self.train_images = torch.from_numpy(self.train_images).unsqueeze(1).float() / 255
        self.test_images = torch.from_numpy(self.test_images).unsqueeze(1).float() / 255

    def __len__(self):
        return len(self.test_images if self.isTest else self.train_images)

    def __getitem__(self, id):
        if self.isTest:
            return ministTransforms(self.test_images[id]), self.test_labels[id]
        else:
            return ministTransforms(self.train_images[id]), self.train_labels[id]
