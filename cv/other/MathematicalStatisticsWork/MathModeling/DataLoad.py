import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import DataProcess

input_dim = 367


class GasDataset(Dataset):
    def __init__(self, dataPath=DataProcess.dataDir + r'\附件一：325个样本数据.xlsx'):
        self.dataPath = dataPath
        df = pd.read_excel(dataPath)
        sourceData = df.to_numpy()
        sourceData = np.delete(sourceData, (0, 1), axis=1)  # 先删除不需要的行和列
        property_name_list = sourceData[0, :]  # 取出属性名字，后面用到
        self.trainData = np.delete(sourceData, (0, 1), axis=0).astype(np.float)
        self.trainlabel = self.trainData[:, 9]  # 第9列是回归值，标签
        self.trainData = np.delete(self.trainData, 9, axis=1)
        for i in range(self.trainData.shape[1]):
            col = self.trainData[:, i]
            num_0 = np.sum(col == 0)  # 0的数量
            if num_0 == col.size:
                print('第%d列全为0' % i)
        self.trainData = self.trainData / self.trainData.max(axis=0)
        self.trainData = torch.tensor(self.trainData, dtype=torch.float)
        self.trainlabel = torch.tensor(self.trainlabel, dtype=torch.float).reshape((325, 1))
        self.nothing = 1

    def __len__(self):
        return len(self.trainData)
        # return 2000

    def __getitem__(self, id):
        return self.trainData[id], self.trainlabel[id]
        # x = torch.randn(input_dim)
        # return x, x.dot(x)


if __name__ == '__main__':
    GasDataset()
