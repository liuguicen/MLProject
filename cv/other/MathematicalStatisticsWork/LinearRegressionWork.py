import matplotlib.pyplot as plt
import numpy as np

x = [93,
97,
99,
100,
92,
94,
94,
97,
84,
86,
88,
91,
74,
77,
81,
86]
y = [30, 35, 40, 45, 50, 55, 66, 75, 85, 100, 110, 120, ]

x = np.array(x)
y = np.array(y)

x_mean = x.mean()
y_mean = y.mean()
print("样本和= ", x.sum(), "样本数量 = ", x.size)
print("X样本均值= ", x.mean(), "x样本方差 ", x.var())
print('Y样本均值= ', y.mean())
#
# sxy = 0
# for i, xi in enumerate(x):
#     sxy += (xi - x_mean) * (y[i] - y_mean)
# sxx = (x.var() * len(x))
# syy = (y.var() * len(y))
#
# b = sxy / sxx
# a = y_mean - b * x_mean
#
# y_pre = []
# for xi in x:
#     y_pre = a * x + b
#
# print('b = ', b)
# print('a = ', a)
# print("回归标准差值 = ", np.sqrt(1 / (len(x) - 2) * (syy - b * sxy)))
# print(a * 4.2 + b)
