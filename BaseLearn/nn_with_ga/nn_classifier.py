import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import cart10_data
import minist_data

super_para_min = [1, 1, 1, 1, 1]
# 超参数二进制位长度
super_parameter_bin_len = [3, 8, 9, 9, 9]


class Net(nn.Module):
    def __init__(self, super_para):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, super_para[1], super_para[0])  # 二维的卷积，这两个属性应该是自带的，
        # 还可以使用add_module将添加到额外的层到网络中。
        # 如
        # self.add_module("maxpool1", nn.MaxPool2d(2, 2))
        # self.add_module("covn3", nn.Conv2d(64, 128, 3))
        # self.add_module("conv4", nn.Conv2d(128, 128, 3))
        # 在正向传播的过程中可以使用添加时的name来访问改layer。
        # 这就需要使用到torch.nn.ModuleList和torch.nn.Sequential。
        #
        # # 使用ModuleList和Sequential可以方便添加子网络就是多个层到网络中，但是这两者还是有所不同的。
        # ModuleList可以将一个Module的List加入到网络中，自由度较高，但是需要手动的遍历ModuleList进行forward。
        # Sequential按照顺序将将Module加入到网络中，也可以处理字典。 相比于ModuleList不需要自己实现forward
        self.conv2 = nn.Conv2d(super_para[1], super_para[2], super_para[0])
        # 最后的矩阵大小
        last_w = ((32 - super_para[0] + 1) // 2 - super_para[0] + 1) // 2
        self.fc1 = nn.Linear(super_para[2] * last_w * last_w, super_para[3])
        self.fc2 = nn.Linear(super_para[3], super_para[4])
        self.fc3 = nn.Linear(super_para[4], 10)
        # self.fc1 = nn.Linear(16 * 4 * 4, super_para[0])
        # self.fc2 = nn.Linear(super_para[0], super_para[1])
        # self.fc3 = nn.Linear(super_para[1], 10)

    def forward(self, x):
        # relu为激活函数，相当于吴恩达课程中的由z得到a
        # 池化，就是采样，减少“图片”尺寸大小，采用规则应该就是2*2的区域变成1
        # max_pool2d表示取最大值，对应的还有avg_pool等
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 第二层卷积，激活，采样
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 将张量变成一维向量的形式
        x = x.view(-1, self.num_flat_features(x))
        # 这一层的输出，乘上参数矩阵，得到下一层的输入，再经过relu函数，变成下一层的输出
        # 参数矩阵行数等于输入向量的列数，参数矩阵的列数等于输出向量的列数，最后得到输出的一个行向量
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def training(train_data, net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    # print('使用设备', device, '训练')
    net.to(device)
    y_loss = []
    y_total = []
    x_epoch = []
    x_total = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 从实验结果上看，官方预测的两轮就差不多了，更多epoch没用
    for epoch in range(3):
        running_loss = 0.
        for i, data in enumerate(train_data, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print_count = 100
            # if i % print_count == print_count - 1:  # print every 2000 mini-batches
            #     print('[epoch = %d, batch = %5d] average  loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / print_count))
            #     x_epoch.append(i + 1)
            #     y_loss.append(running_loss / print_count)
            #     running_loss = 0.0
            # 少训练
            if i >= 10000:
                print('训练%d批次，退出训练' % 10000)
                break
        # plt.plot(x_epoch, y_loss)
        # x_total.append(x_epoch)
        # y_total.append(y_loss)
        # x_epoch.clear()
        # y_loss.clear()
    # plt.show()
    # plt.plot(x_total, y_total)
    # plt.show()
    return net


def test(net, testloader, classes):
    test_iter = iter(testloader)
    images, labels = test_iter.next()
    minist_data.imshow(torchvision.utils.make_grid(images))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images, labels = images.to(device), labels.to(device)
    # show images
    # print labels
    print(labels)
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outs = net(images)
    _, predicted = torch.max(outs, 1)
    print("predicted:\n ", ' '.join("%5s" % classes[predicted[j]] for j in range(4)))
    test_all(net, testloader, classes)


def test_all(net, test_data, classes):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_data, 0):
            images, labels = data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i >= 100:
                break
    print('Accuracy of the network on the 10000 test images: %f %%' % (
            100 * correct / total))
    return 100 * correct / total


def get_real_para(x):
    return [x[i] + super_para_min[i] for i in range(0, len(x))]


data_processor = None
net = None


def m_test(x):
    print('my test')
    global data_processor
    global net
    x += super_para_min
    global data_processor
    if not data_processor:
        data_processor = cart10_data.DataProcessor()
    try:
        net = torch.load('.\models\model_di111git.pkl')
        print('load net from disk')
    except:
        net = Net(x)
        net = training(data_processor.train_data, net)
        # 保存
        # net = Net()
        torch.save(net, '.\models\model_digit.pkl')
        print('training and save net')

    test_all(net, data_processor.verify_data, data_processor.classes)


def run(x):
    # return x[0] + 1/(x[1] + 0.1) + x[2]
    super_paras = [x[i] + super_para_min[i] for i in range(0, len(x))]
    print('超参数为', super_paras)
    global data_processor
    if not data_processor:
        data_processor = cart10_data.DataProcessor()
    net = Net(super_paras)
    net = training(data_processor.train_data, net)
    # test(net, data_processor.test_data, data_processor.classes)
    accuracy_rate = test_all(net, data_processor.verify_data, data_processor.classes)
    return accuracy_rate


def finish():
    torch.save(net, '.\models\model_digit.pkl')
    print('training and save net')


if __name__ == '__main__':
    m_test([5, 255, 400, 331, 285])
    m_test([5, 255, 400, 331, 285])
    m_test([5, 255, 400, 331, 285])
