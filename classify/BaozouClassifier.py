import random

import path
from torchvision.models import MobileNetV3
import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, models, transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ml_base import CommonConstantParam

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 10  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.0001  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(0.5),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.RandomVerticalFlip(0.5),  # 竖直翻转
    transforms.RandomRotation(30),
    transforms.RandomCrop(128, padding=4),
    #     transforms.ColorJitter(brightness=0.5),
    #     transforms.ColorJitter(contrast=0),
    transforms.ToTensor(),
    transforms.Normalize(CommonConstantParam.image_net_mean, CommonConstantParam.image_net_std),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(CommonConstantParam.image_net_mean, CommonConstantParam.image_net_std),
])

import shutil


def prePareData(dir):
    # 获取所有文件路径，80训练，20%测试
    train_dir = path.join(dir, 'train')
    test_dir = path.join(dir, 'test')
    FileUtil.mkdir(path.join(dir, 'test'))
    for _, dirList, _ in os.walk(train_dir):
        for classDir in dirList:
            test_class = path.join(test_dir, classDir)
            if path.exists(test_class):
                continue
            FileUtil.mkdir(test_class)
            for root, _, fileList in os.walk(path.join(train_dir, classDir)):
                random.shuffle(fileList)
                for name in fileList[0:int(len(fileList) * 0.2)]:
                    res = shutil.move(os.path.join(train_dir, classDir, name),
                                      os.path.join(test_dir, classDir, name))
                    print('移动图片' + res + '成功')


# 将数据放到TrainLoader和TestLoader中
class MyDataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
        self.imgs = []
        for _, classDirList, _ in os.walk(dataset_path):
            for id, className in enumerate(classDirList):
                for _, _, fileList in os.walk(path.join(dataset_path, className)):
                    for fileName in fileList:
                        if fileName.endswith('jpg') or fileName.endswith('jpeg') or fileName.endswith(
                                'png') or fileName.endswith('webp') or fileName.endswith('gif'):
                            self.imgs.append((path.join(dataset_path, className, fileName), id))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def createDataLoader(dir, isTest):
    datasets = MyDataset(dir, transform=transform_train if isTest else transform_test)
    # 由于我使用的是Win10系统，num_workers只能设置为0，其他系统可以调大此参数，提高训练速度
    testloader = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return testloader


def showImg(dataLoader):
    # 查看图片的代码，不执行不会影响后续的训练
    to_pil_image = transforms.ToPILImage()
    cnt = 0
    for image, label in dataLoader:
        if cnt >= 3:  # 只显示3张图片
            break
        print(label)  # 显示label

        img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
        img = img.numpy()  # FloatTensor转为ndarray
        img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后

        # 显示图片
        plt.imshow(img)
        plt.show()
        cnt += 1


# MobileNet V2的预训练模型
class MobileNet(nn.Module):
    def __init__(self, num_classes=3):  # num_classes，此处为 二分类值为2
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)  # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(  # 定义自己的分类层
            nn.Linear(1280, 1000),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(True),
            nn.Dropout(0.5),
            #                 nn.Linear(1024, 1024),
            #                 nn.ReLU(True),
            #                 nn.Dropout(0.3),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 定义两个函数，一个可以冻住features层，只训练FC层，另一个把features层解冻，训练所有参数
from collections.abc import Iterable


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


# 定义两个数组，为了存储预测的y值和真实的y值
y_predict = []
y_true = []
# 我不导入这个包会报错，
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ml_base.common_lib_import import *


# 训练
def train():
    net = MobileNet().to(device)

    # 选择优化器和Loss
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    trainloader = createDataLoader(path.join(dataset_dir, "train"), False)
    testloader = createDataLoader(path.join(dataset_dir, "test"), True)
    print("Start Training!")  # 定义遍历数据集的次数
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # 准备数据
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            # 使用Top5分类
            maxk = max((1, 1))
            label_resize = labels.view(-1, 1)
            _, predicted = outputs.topk(maxk, 1, True, True)
            total += labels.size(0)
            correct += torch.eq(predicted, label_resize).cpu().sum().float().item()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)

                maxk = max((1, 1))
                label_resize = labels.view(-1, 1)
                _, predicted = outputs.topk(maxk, 1, True, True)
                total += labels.size(0)
                correct += torch.eq(predicted, label_resize).cpu().sum().float().item()

                y_predict.append(predicted)
                y_true.append(labels)
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
    torch.save(net, './model/mobileNet_freeze.pth')


dataset_dir = "/home/lgc/桌面/E/dataset/BaozouClassify"
if __name__ == "__main__":
    prePareData(dataset_dir)
    train()
    print('start')
