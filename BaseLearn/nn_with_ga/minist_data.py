# 通常来说，当你处理图像，文本，语音或者视频数据时，你可以使用标准 python 包将数据加载成 numpy 数组格式，然后将这个数组转换成 torch.*Tensor
# 对于图像，可以用 Pillow，OpenCV
# 对于语音，可以用 scipy，librosa
# 对于文本，可以直接用 Python 或 Cython 基础数据加载模块，或者用 NLTK 和 SpaCy

import torch
import torchvision
import torchvision.transforms as transforms

print(__name__)


class MnistDataProcessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.classes = ('0', '1', '2', '3', '4',
                        '5', '6', '7', '8', '9')
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=False, num_workers=2)
        self.train_data = []
        self.verify_data = []
        for i, data in enumerate(trainloader, 0):
            if i > 5000:
                self.verify_data.append(data) #从训练集中取交叉验证集
            if i > 7000:
                break
            self.train_data.append(data)


import matplotlib.pyplot as plt
import numpy as np


def test():
    data_processor = MnistDataProcessor()
    trainloader, testloader, classes = data_processor.train_loader, data_processor.test_loader, data_processor.classes
    # functions to show an image

    # get some random training images
    for i, data in enumerate(testloader, 0):
        images, labels = data

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(labels)
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
        break


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test()
    test()
