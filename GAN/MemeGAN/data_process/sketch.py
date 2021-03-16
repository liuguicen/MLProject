# -*- coding: utf-8 -*-

import cv2

from ml_base import fileUtil


def useCanny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


import os
import imgBase
import numpy as np
import edge.Hed_edge_detect as hed
import json


def getEdge(img_path):
    img = imgBase.cv_imread_CN(img_path)  # type:np.ndarray
    # 将图片的透明背景改成白色的，因为透明背景会被当成黑色处理，实际上白色才更符合
    img = imgBase.transparence2white(img)
    img_w, img_h = img.shape[0], img.shape[1]
    if img_w < 64 or img_h < 64:  # 尺寸小的时候，hed无法检测出来，canny可以，单纯放大图像再使用hed也不行
        ratio = max(128 / img_w, 128 / img_h)
        img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        edges = useCanny(img)
        # 加个平滑可能更好
    else:
        edges = hed.detectEdge(img)
        edges = np.uint8(edges * 255)
    edges = imgBase.colorConvert(edges, edges)  # 颜色反转，使用黑色表示边缘，原来的算法输出的是白色表示边缘
    # cv2.imshow('edges', edges)
    return img, edges
    # cv2.waitKey(0)


#
# edges = getEdge('F:\\重要_data_set__big_size\\表情\\大黄脸\\bsc.webp')
# cv2.imshow('edges', edges)
# cv2.waitKey(0)
# exit()

run_record_path = 'run_record.txt'
emoji_edge_id_key = 'emoji_edge_id'


def get_record():
    if not os.path.exists(run_record_path):
        fp = open(run_record_path, 'w')
        fp.close()
    run_record = json.load(open(run_record_path, 'r'))  # type:dict
    emoji_edge_id = run_record.get(emoji_edge_id_key)
    return run_record, emoji_edge_id


def make_edge_map(run_record, emoji_edge_id):
    src_dir, edge_dir = None, None
    for root, dirs, files in os.walk('F:\\重要_data_set__big_size\\表情\\大黄脸\\'):
        # 创建目录
        preprocessDir = os.path.join(root, '预处理后的')
        src_dir = os.path.join(preprocessDir, 'src')
        edge_dir = os.path.join(preprocessDir, '边缘图')
        if not os.path.exists(src_dir):
            os.makedirs(src_dir)
        if not os.path.exists(edge_dir):
            os.makedirs(edge_dir)

        for id, file_name in enumerate(files):
            if id <= emoji_edge_id:
                continue
            if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith(
                    'jpeg') or file_name.endswith('webp'):
                img_path = os.path.join(root, file_name)
                src_img, edges = getEdge(img_path)

                # 保存预处理后的图片
                src_path = os.path.join(src_dir, file_name.split('.')[0] + '.jpg')
                edge_path = os.path.join(edge_dir, file_name.split('.')[0] + '.jpg')

                imgBase.cv_imwrite_CN(src_path, src_img)
                imgBase.cv_imwrite_CN(edge_path, edges)
                print(file_name, id, 'make edge finish')
            run_record[emoji_edge_id_key] = id
        break  # 只处理一级目录
    return src_dir, edge_dir


import random
import shutil
import logging


def split_train_test(a_src_dir, b_src_dir, dst_dir):
    '''
    :param a_src_dir: a类图像父目录
    '''

    def prapare(path):
        trainPath = os.path.join(path, 'train')
        fileUtil.mkdir(trainPath)
        valPath = os.path.join(path, 'val')
        fileUtil.mkdir(valPath)
        testPath = os.path.join(path, 'test')
        fileUtil.mkdir(testPath)
        return trainPath, valPath, testPath

    def copyFile(src_dir, srcNameList, dst_dir):
        for name in srcNameList:
            try:
                shutil.copy(os.path.join(src_dir, name), os.path.join(dst_dir, name))
                print('拷贝', name, '成功')
            except:
                logging.error('拷贝 ', name, ' 失败')
                continue

    atrainDir, avalDir, atestDir = prapare(os.path.join(dst_dir, 'A'))
    btrainDir, bvalDir, btestDir = prapare(os.path.join(dst_dir, 'B'))

    nameList = os.listdir(a_src_dir)
    random.shuffle(nameList)
    train_count = int(len(nameList) * 0.8)
    copyFile(a_src_dir, nameList[0: train_count], atrainDir)
    copyFile(a_src_dir, nameList[train_count: len(nameList)], atestDir)

    copyFile(b_src_dir, nameList[0:train_count], btrainDir)
    copyFile(b_src_dir, nameList[train_count: len(nameList)], btestDir)


if __name__ == '__main__':
    run_record, has_processed_id = get_record()
    try:
        aPath, bPath = make_edge_map(run_record, has_processed_id)
        split_train_test(aPath, bPath, os.path.dirname(aPath))
    except Exception as e:
        print('异常中断，写入中断位置')
        json.dump(run_record, open(run_record_path, 'w'))
        raise e
