import torch
import torch.nn as nn

# BatchNorm2D动量参数 原论文实现用的这个
BN_MOMENTUM = 0.2


# 占位符
# f(x) = x
class PlaceHolder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class HRNetConv3x3(nn.Module):

    def __init__(self, intc, outc, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(intc, outc, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(outc, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class HRNetStem(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        # 通道数量变为1/2
        self.conv1 = HRNetConv3x3(inc, outc, stride=2, padding=1)
        self.conv2 = HRNetConv3x3(outc, outc, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


# 先经过stem，再转换通道数匹配stage1的输入通道数
class HRNetInput(nn.Module):
    def __init__(self, inc, outc, stage1_inc):
        super().__init__()
        self.stem = HRNetStem(inc, outc)
        self.in_change_conv = nn.Conv2d(outc, stage1_inc, kernel_size=1, stride=1, bias=False)
        self.in_change_bn = nn.BatchNorm2d(stage1_inc, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.stem(x)
        x = self.in_change_conv(x)
        x = self.in_change_bn(x)
        x = self.relu(x)
        return x


class NormalBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv1 = HRNetConv3x3(inc, outc, padding=1)
        self.conv2 = HRNetConv3x3(outc, outc, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv1 = HRNetConv3x3(inc, outc, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outc, momentum=BN_MOMENTUM)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = self.relu2(x)

        return x


class HRNetStage(nn.Module):
    def __init__(self, stagec_list, block):
        super().__init__()
        # stagec 是列表，表示的是多个分支的通道数量
        self.stagec_list = stagec_list
        self.stage_branch_num = len(stagec_list)
        self.block = block
        self.block_num = 4
        self.stage_layers = self.create_stage_layers()

    def create_stage_layers(self):
        tostage_layers = []  # 并行的分支
        for i in range(self.stage_branch_num):
            branch_layer = []  # 分支里面的所有层
            for j in range(self.block_num):
                branch_layer.append(self.block(self.stagec_list[i], self.stagec_list[i]))
            branch_layer = nn.Sequential(*branch_layer)
            tostage_layers.append(branch_layer)
        return tostage_layers

    def forward(self, x_list):
        out_list = []
        for i in range(self.stage_branch_num):
            out_list.append(self.stage_layers[i](x_list[i]))
        return out_list


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = HRNetInput(3, outc=64, stage1_inc=32)
        self.stage1 = HRNetStage([32], ResBlock)

    def forward(self, x):
        x = self.input(x)
        return x


if __name__ == "__main__":
    model = TestNet()
    data = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.float32)
    y = model(data)
    print(model)
    print(y)
