import logging
import os
import pickle


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def writeList(list, path):
    '''
    path txt文件
    '''
    fileObject = open(path, 'w')
    for item in list:
        fileObject.write(str(item))
        fileObject.write('\n')
    fileObject.close()


def readList(path):
    list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            list.append(line.strip('\n'))
    return list


if __name__ == '__main__':
    print(os.path.dirname("abbdd/ddd/xdx/t.txt"))


def getChildPath_firstLeve(path):
    child_list = []
    for root, dir_name, file_list in os.walk(path):
        for file_name in file_list:
            child_list.append(os.path.join(root, file_name))
        return child_list


from os import path


def getFileName(pathObj):
    if isinstance(pathObj, str):
        return path.splitext(path.basename(pathObj))[0]
    return None


def getChildPath_AllLeve(dir, suffix: str):
    child_list = []
    for root, dir_name, file_list in os.walk(dir):
        for file_name in file_list:  # type:str
            if suffix is None or file_name.endswith(suffix):
                child_list.append(os.path.join(root, file_name))
    return child_list

def getChildFirst():
    '''
    os.listdir
    '''
    pass

def getAllChild():
    '''
    os.walk
    '''
    pass