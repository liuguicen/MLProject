# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:07:23 2019
@author: AugustMe
"""
import numpy as np
import os
import gzip

import torch.utils.data

wikiartPath = r'E:\重要_dataset_model\wikiart\train'
animatePath = r'E:\重要_dataset_model\动画漫画\动画漫画'
cocoPath = r'E:\重要_dataset_model\COCO\train2014'

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

    def __init__(self):
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load_data(
            r'E:\重要_dataset_model\MNIST/')

    def __len__(self):
        return len(self.test_images if self.isTest else self.train_images)

    def __getitem__(self, id):
        if self.isTest:
            return self.test_images[id], self.test_labels[id]
        else:
            return self.train_images[id], self.train_labels[id]
