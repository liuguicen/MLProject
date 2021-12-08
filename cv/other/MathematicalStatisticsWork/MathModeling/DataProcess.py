import numpy  as np
import pandas as pd


def readOperVar():
    # 读取操作范围
    dataPath = dataDir + r'\附件四：354个操作变量信息.xlsx'
    df = pd.read_excel(dataPath)
    srcData = df.to_numpy()
    srcData = np.delete(srcData, 0, axis=1)
    propertyDict = {}
    # 数据解析
    for row in srcData:
        range = row[2]
        # 删掉括号
        range = range.replace('(', '')
        range = range.replace(')', '')
        range = range.replace('（', '')
        range = range.replace('）', '')
        # 找到破折号出现的位置，至少第二个位置，排除负数的负号
        id = str(range).find('-', 1)
        low, high = range[0:id], range[id + 1:]
        propertyDict[row[0]] = (float(low), float(high), float(row[4]))
        # print('特性', row[0], '值为', propertyDict[row[0]])
    return propertyDict


def readOperVar1():
    dataPath = dataDir + r'\附件一：325个样本数据.xlsx'
    df = pd.read_excel(dataPath)
    sourceData = df.to_numpy()
    sourceData = np.delete(sourceData, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), axis=1)  # 先删除不需要的行和列
    property_name_list = sourceData[0, :]  # 取出属性名字，后面用到
    sourceData = np.delete(sourceData, (0, 1), axis=0).astype(np.float)
    sourceData = np.transpose(sourceData)
    propertyDict = {}
    for i, name in enumerate(property_name_list):
        propertyDict[name] = (np.min(sourceData[i]), np.max(sourceData[i]))
        # print(data_1.shape, data_2.shape)
    return propertyDict


def del_loss(data, propertyDict, property_name_list):
    new_data = []
    # 第一步，（1） 和 （2）要求，
    row_num = data.shape[0]
    col_num = data.shape[1]
    loss_threshold = row_num / 2  # 残缺的阈值
    remove = set()
    for i in range(col_num):
        col = data[:, i]
        num_0 = np.sum(col == 0)  # 0的数量
        mean = np.mean(col)

        propertyRange = propertyDict.get(property_name_list[i], None)
        if propertyRange is None:
            print("属性", property_name_list[i], '无法找到取值范围等信息')
            propertyRange = (-float("inf"), float("inf"), 0.00001)

        # 处理缺失的值
        # 有取值为0的样本并且取值范围不能是0
        if num_0 > 0:
            print('第', i, '列出现为0的元素%d个'%num_0)
            if not (propertyRange[0] <= 0 and propertyRange[1] >= 0):
                print('第%d列等于0的元素数量= %d' % (i, num_0))
                if num_0 < loss_threshold:
                    print('未超过阈值%f，将其填充为均值%f' % (loss_threshold, mean))
                    col[col == 0] = mean  # 要求3，缺失数据比较少时，缺失数据变成均值
                if num_0 > loss_threshold:
                    print('超过阈值 %f 删除该列数据点' % (loss_threshold))
                    col[col != 0] = 0  # 要求1,2 ， 删除残缺数据较多的点以及全为空值的点，这里面就是0的个数大于某个阈值的列删掉

        # 纠正不在最大最小范围内的样本
        # 文档里面说剔除样本，但是貌似所有样本都会被剔除，这里超过之后直接改为最大/最小值
        # 处理超过最大-最小范围的值
        for j in range(row_num):
            # 空数据，跳过
            if col[j] == 0:
                continue

            if col[j] < propertyRange[0]:
                print("第%d列%d行的元素值为%f, 小于最小值%f，纠正" % (i, j, col[j], propertyRange[0]))
                col[j] = propertyRange[0]
                if i != 48:
                    remove.add(j)
            if propertyRange[1] < col[j]:
                print("第%d列%d行的元素值为%f, 大于最大值%f，纠正" % (i, j, col[j], propertyRange[1]))
                col[j] = propertyRange[1]
                if i != 48:
                    remove.add(j)

        # 处理超过3倍样本方差的值
        mean = np.mean(col)  # col已经改变，要重新计算
        res_error = col - mean
        sample_var = np.std(col, ddof=1)  # 样本标准差
        # print('原值', col, '均值', mean, '剩余误差', res_error, '样本标准差', sample_var)
        for j in range(row_num):
            if col[j] < mean - sample_var * 3:
                print("第%d列%d行的元素值为%f, 小于3倍样本方差%f，纠正" % (i, j, col[j], mean - sample_var * 3))
                col[j] = mean - sample_var * 3
                remove.add(j)
            if col[j] > mean + sample_var * 3:
                print("第%d列%d行的元素值为%f, 超过3倍样本方差%f，纠正" % (i, j, col[j], mean + sample_var * 3))
                col[j] = mean + sample_var * 3
                remove.add(j)

        new_data.append(col)
    print('删除的行', list(remove))
    new_data = np.array(new_data)
    new_data = np.transpose(new_data)
    return new_data


def getFinnalRes(data):
    res = []
    for i in range(data.shape[1]):
        col = data[:, i]
        res.append(np.mean(col))
    return res


def writeData(data):
    data = pd.DataFrame(data)
    with pd.ExcelWriter('数据预处理结果.xlsx', mode='w') as writer:  # 写入Excel文件
        data.to_excel(writer, float_format='%.5f')  # ‘page_1’是写入excel的sheet名


dataDir = r'E:\读研相关\其它重要的\建模\2020年中国研究生数学建模竞赛赛题\2020年B题\2020年B题--汽油辛烷值建模\数模题'


def process():
    dataPath = dataDir + r'\附件三：285号和313号样本原始数据.xlsx'
    df = pd.read_excel(dataPath, sheet_name='操作变量')
    sourceData = df.to_numpy()
    sourceData = np.delete(sourceData, (0,), axis=1)  # 先删除第一列，注意顺序不能换
    property_name_list = sourceData[0, :]  # 取出属性名字，后面用到
    sourceData = np.delete(sourceData, (0, 1, 42), axis=0).astype(np.float)

    # 两组数据，分别切分出来
    data_1 = sourceData[0:40, :]
    data_2 = sourceData[41:80, :]

    propertyDict = readOperVar1()
    print('处理1')
    data_1 = del_loss(data_1, propertyDict, property_name_list)
    print('处理2')
    data_2 = del_loss(data_2, propertyDict, property_name_list)

    print('最后结果')
    data_1, data_2 = np.mean(data_1, axis=0), np.mean(data_2, axis=0)

    property_name_list = property_name_list.reshape(1, property_name_list.size)
    data_1 = data_1.reshape(1, data_1.size)
    data_2 = data_2.reshape((1, data_2.size))

    res = np.concatenate((property_name_list, data_1, data_2), axis=0)
    writeData(res)


if __name__ == '__main__':
    process()
# print(cel)
# print(npData)
# print(df.size)
# print(df)
