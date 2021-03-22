import torch

a = torch.rand(5, 5, 3)
print(a.size())
b = a.permute(1, 2, 0)
print(a.size(), b.size())
