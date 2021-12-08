import torch
#
# x = torch.rand(5, 3)
# print(x, x.size())
# print(x[1, :])
# x = torch.rand(4, 4)
# print(x.view(16).size(), x.view(-1, 8).size())
# # is_available 函数判断是否有cuda可以使用
# # ``torch.device``将张量移动到指定的设备中
# if torch.cuda.is_available():
#     print("cuda is available")
#     device = torch.device("cuda")          # a CUDA 设备对象
#     y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
#     x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改
print(torch.ones(2))