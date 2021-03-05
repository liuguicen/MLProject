import torch

x = torch.ones(2, requires_grad=True)
y = (x * x * 4)

y.backward(gradient=torch.tensor(1))

print('x=', x, 'y = ', y)
print('x.grad', x.grad)
