import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# x = np.linspace(0, 20 * np.pi, 5)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()
# plt.subplot(231)
# x = np.linspace(0, 20 * np.pi, 5)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()
from matplotlib import pyplot as plt
import networkx as nx

# graph = [[0, 6.5, 4, 10, 5, 7.5, 11, 10, 4],
#          [6.5, 0, 7.5, 10, 10, 7.5, 7.5, 7.5, 6],
#          [4, 7.5, 0, 10, 5, 9, 9, 15, 7.5],
#          [10, 10, 10, 0, 10, 7.5, 7.5, 10, 9],
#          [5, 10, 5, 10, 0, 7, 9, 7.5, 20],
#          [7.5, 7.5, 9, 7.5, 7, 0, 7, 10, 10],
#          [11, 7.5, 9, 7.5, 9, 7, 0, 10, 16],
#          [10, 7.5, 15, 10, 7.5, 10, 10, 0, 8],
#          [4, 6, 7.5, 9, 20, 10, 16, 8, 0]]
graph = [[0,1,1,1],[1,0,2,2],[1,1,0,2],[1,1,2,0]]
mt = np.matrix(graph)
g = nx.from_numpy_matrix(mt)
# g.add_nodes_from([1,2,3])
# g.add_edges_from([(1,2),(1,3)])
nx.draw_networkx(g)
plt.show()
