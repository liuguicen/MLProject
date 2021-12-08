import csv
import time

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

app_data_path = '.\\data\\app数据.csv'


app_feature_list = []



def data2time(data):
    # 字符类型的时间
    # 转为时间数组
    timeArray = time.strptime(data, "%Y/%m/%d")
    # print(timeArray)
    # timeArray可以调用tm_year等
    # print(timeArray.tm_year)  # 2013
    # 转为时间戳
    return float(time.mktime(timeArray)) / 24 / 3600


def showDau(time_list, dau_list):
    new_dau = []
    id = 0
    for t, a in zip(time_list, dau_list):
        new_dau.append(a)
        print('id =', id, 'dau = ', a)
        id += 1
    plt.plot(new_dau)
    plt.show()


def read_and_vec_app_feature():
    dau_list = []
    time_list = []
    picn_list = []
    coden_list = []
    holiday_list = []
    with open(app_data_path, 'r') as movie_data_file:
        csv_reader = csv.reader(movie_data_file)
        for i, row in enumerate(csv_reader, 0):
            if i >= 1:
                # print(row)
                holiday = float(row[4])
                holiday_list.append(holiday)

                dau = [float(row[0]) / holiday ]
                dau_list.append(dau)

                time = data2time(row[1])
                time_list.append(time)

                pic_n = float(row[2])
                picn_list.append(pic_n)

                code_n = float(row[3])
                coden_list.append(code_n)

                print('dau', dau, 'time', time, '图片数量', pic_n, '代码数量', code_n
                      , '节假日系数', holiday)

    # print(dau_list)
    # print(time_list)
    # print(coden_list)

    time_list = normalize(np.array(time_list))
    coden_list = normalize_1(np.array(coden_list))
    picn_list = normalize_1(np.array(picn_list))
    # holiday_list = normalize_1(np.array(holiday_list))
    dau_list = normalize(np.array(dau_list))
    # showDau(time_list, dau_list)
    # plt.plot(time_list)
    plt.plot(dau_list)
    plt.plot(picn_list)
    plt.show()
    # font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
    #
    # plt.xlabel(u"时间", fontproperties=font)
    # plt.ylabel(u"图片数量", fontproperties=font)
    # plt.plot(picn_list)
    # plt.plot(coden_list)
    # plt.plot(holiday_list)
    # plt.show()

    x = []
    y = dau_list
    # print('归一化之后')
    for time, pic, code in zip(time_list, picn_list, coden_list):
        x.append([time, pic, code])

    # print('time', time, '节假日系数', holiday, "日活", dau)
    return x, y


x, y, mid = None, None, -1


def get_next_batch(batch):
    global x, y, mid
    if x is None:
        x, y = read_and_vec_app_feature()
    return torch.tensor(x), torch.tensor(y)
    # mid = (mid + 1) % (len(x) - 1)
    # return torch.tensor(x[mid * batch: (mid + 1) * batch]), torch.tensor(y[mid * batch:(mid + 1) * batch])


def normalize(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)


def normalize_1(data):
    max = np.amax(data)
    return data / max


if __name__ == '__main__':
    x_batch, y_batch = get_next_batch(4)
    print(x_batch)
    print(x_batch.size())
    print(y_batch)
    print(y_batch.size())