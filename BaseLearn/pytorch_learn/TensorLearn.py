import torch
from torch import Tensor
from torch import tensor

a = tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
b = tensor([[3, 4], [5, 6]])
c = a.mul(b).sum()
print(c)
c.backward()
print(a.grad)
print(torch.rand((3,4)))
print(torch.randn((3,4)))