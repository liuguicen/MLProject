from torchvision.models import mobilenet

net = mobilenet.MobileNetV2()
import torch

print(torch.tensor([[1, 1, 1], [2, 2, 2]]) * torch.tensor([[2, 2], [4, 4]]))
