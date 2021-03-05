import torch
print(torch.cuda.device_count())
print(hasattr(torch.tensor([]), 'stride'))