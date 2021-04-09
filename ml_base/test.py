from torchvision.models import mobilenet

net = mobilenet.MobileNetV2()
import torch


def m(a, b=1):
    print(a, b)


m(3, b=None)
