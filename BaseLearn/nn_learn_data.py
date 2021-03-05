# 通常来说，当你处理图像，文本，语音或者视频数据时，你可以使用标准 python 包将数据加载成 numpy 数组格式，然后将这个数组转换成 torch.*Tensor
# 对于图像，可以用 Pillow，OpenCV
# 对于语音，可以用 scipy，librosa
# 对于文本，可以直接用 Python 或 Cython 基础数据加载模块，或者用 NLTK 和 SpaCy

import torch
import torchvision
import torchvision.transforms as transforms

print(__name__)


class DataProcessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_data = self.load_training_data()
        self.test_data = self.load_test_data()

    def load_training_data(self):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=2)
        return trainloader

    def load_test_data(self):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)
        return testloader


import matplotlib.pyplot as plt
import numpy as np


def test():
    data_processor = DataProcessor()
    trainloader, testloader, classes = data_processor.train_data, data_processor.test_data, data_processor.classes
    # functions to show an image

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(labels)
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test()
