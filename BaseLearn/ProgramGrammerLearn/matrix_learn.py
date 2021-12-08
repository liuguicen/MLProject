import numpy as np

a = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 1, 1, 0])
print(a)
print(a.T)
b = np.matmul(a.T, a)
print(b)
c = np.linalg.pinv(b)
print(c)
d = np.matmul(c.T, a.T)
print(d)
w = np.matmul(d, y)
print(w)
print(np.matmul(np.array([0, 1, 1]), w))


