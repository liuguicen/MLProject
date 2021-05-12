import logging
import os
import pickle


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def saveRunRecord(obj, path):
    '''
    保存运行记录，直接用下面的代码就行，写在这里防止忘记
    '''
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def readRunRecord(path):
    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)
    else:
        print('没有运行记录')
