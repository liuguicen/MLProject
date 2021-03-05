print(max([1, 2], key=lambda a: 1 / a))


def a():
    b()


def b():
    pass


import matplotlib
from matplotlib import pyplot as plt

font = {
    'family': 'SimHei',
    'weight': 'bold',
    'size': 12
}
matplotlib.rc("font", **font)

_x_ticks = [20170912, 201701913, 201701914, 201701915, 201701916, 201701917, 201701918]
x = [1, 2, 3, 4, 5, 6, 7]
# y1 = [0.001, 0.0035, 0.0095, 0.001, 0.0015, 0.0012, 0.0033]
# y2 = [0.011, 0.006, 0.0165, 0.0055, 0.0115, 0.053, 0.008]
y1 = [0.001, 0.0035, 0.0095, 0.001, 0.0015, 0.0012, 0.0033]
y2 = [0.0165, 0.0125, 0.011, 0.01, 0.0045, 0.0215, 0.018]

line1, = plt.plot(x, y1, linestyle='--', label="RBF模型预测误差率")
line2, = plt.plot(x, y2)
plt.legend(handles=[line2, line1], labels=["RBF模型预测误差率", "GA-RBF模型预测误差率"], loc="upper left", fontsize=6)
plt.xticks(x, _x_ticks)
plt.ylim(0, 0.06)
plt.xlabel("日期")
plt.ylabel('误差率')
plt.title('RBF模型与GA-RBF模型预测误差率曲线')
plt.grid(True, ls='-.', color='r', lw='1')
plt.show()
