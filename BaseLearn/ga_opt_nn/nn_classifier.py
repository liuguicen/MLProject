import numpy as np
from collections import Counter
from sklearn import datasets
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import torch
import numpy
import matplotlib.pyplot as plt

import m_util


# 使用莺尾花iris数据集分类
# 数据准备，
def load_data():
    dataset = datasets.load_iris()
    input = torch.FloatTensor(dataset['data'])
    label = torch.LongTensor(dataset['target'])
    all = []
    for i, x in enumerate(input):
        all.append([x, label[i]])

    # for x in all:
    #     print(x)
    # 共150条数据，洗牌，然后分成，训练数据70，测试数据80
    random.shuffle(all)
    # for x in all:
    #     print(x)
    train = all[:90]
    test = all[90:]
    train_input, train_label = [], []
    test_input, test_label = [], []
    for x in train:
        train_input.append(x[0].numpy().tolist())
        train_label.append(x[1])
    for x in test:
        test_input.append(x[0].numpy().tolist())
        test_label.append(x[1])

    train_input = torch.tensor(train_input)
    train_label = torch.tensor(train_label)

    test_input = torch.tensor(test_input)
    test_label = torch.tensor(test_label)

    # print(train_input)
    # print(train_label)
    return train_input, train_label, test_input, test_label


# 定义BP神经网络
# 三层的bp，可以传入第二层的节点数，默认为12
# 试过4层，效果不如3层的好，应该是过拟合了
class MyNet(torch.nn.Module):
    def __init__(self, super_para):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(4, super_para[0])
        self.fc2 = nn.Linear(super_para[0], 3)
        self.loss_func = torch.nn.CrossEntropyLoss()  # 针对分类问题的损失函数
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.02)  # SGD: 各层神经网络的节点数量
        self.fn = [4, 12, 3]
        # 算出遗传算法向量转为nn参数需要的一些量，
        # 各层权重参数数量 i层节点数 * i+1层的
        self.fwn = [self.fn[i] * self.fn[i + 1] for i in range(len(self.fn)) if i < len(self.fn) - 1]
        print(self.fwn)
        # 各层偏置项数量
        self.fbn = self.fn[1:]
        # 参数总数
        self.para_size = sum(self.fwn) + sum(self.fbn)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_paras(self, para):
        """
        将参数设置给神经网络
        """
        w_list, bias_list = self.vec2para(para)
        self.fc1.weight = w_list[0]
        self.fc1.bias = bias_list[0]
        self.fc2.weight = w_list[1]
        self.fc2.bias = bias_list[1]

    def vec2para(self, paras):
        """
        向量转换成每一层的权重和偏置参数
        """
        start = 0
        w_list = []
        bias_list = []
        for i, wn in enumerate(self.fwn, 0):
            # print("第%d层的权重参数为" % (i + 1))
            w = torch.nn.Parameter(torch.tensor(paras[start:start + self.fwn[i]]).view(-1, self.fn[i]))
            w_list.append(w)
            # print(w)
            start += self.fwn[i]
            # print("\n the bias parameters of the %d layer is " % (i + 1))
            bias = torch.nn.Parameter(torch.tensor(paras[start: start + self.fbn[i]]))
            bias_list.append(bias)
            # print(bias)
            start += self.fbn[i]
        return w_list, bias_list


# 训练数据
def training(net, train_input, train_label):
    y = []
    for i in range(5):
        for t in range(1000):
            out = net(train_input)  # input x and predict based on x
            loss = net.loss_func(out, train_label)  # 输出与label对比
            net.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            net.optimizer.step()  # apply gradients
            # print(loss.item())
            # y.append(loss.item())
    # plt.plot(y)
    # plt.show()


def m_test(net, test_input, test_label):
    out = net(test_input)
    prediction = torch.max(out, 1)[1]  # 1返回index  0返回原值
    pred_y = prediction.data.numpy()
    target_y = test_label.data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    # print("莺尾花预测准确率%.2f" % (accuracy * 100))
    return accuracy


def get_test_accuracy():
    global net, test_input, test_label
    return m_test(net, test_input, test_label)


train_input, train_label, test_input, test_label = None, None, None, None
net = None


def run(paras):
    global net
    if net is None:
        net = MyNet([12])
    global train_input, train_label, test_input, test_label
    if train_input is None:
        train_input, train_label, test_input, test_label = load_data()
    # training(net, train_input, train_label)
    net.set_paras(paras)

    out = net(train_input)  # input x and predict based on x
    loss = net.loss_func(out, train_label)  # 输出与label对比
    return 1. / (loss.item() + 1)
    # m_util.print_para(net)
    # return m_test(net, test_input, test_label)
    # return np.math.fabs(sum(paras))


if __name__ == '__main__':
    if net is None:
        net = MyNet([16])
    if train_input is None:
        train_input, train_label, test_input, test_label = load_data()
    training(net, train_input, train_label)
    print(m_test(net, test_input, test_label))


def get_para_size():
    global net
    if net is None:
        return 99
    else:
        return net.para_size
