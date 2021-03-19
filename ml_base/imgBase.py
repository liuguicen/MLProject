import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def colorConvert(img: np.ndarray, dst=None):
    '''
    反转图片颜色，结果直接保存在图片中
    '''
    w, h = img.shape[0], img.shape[1]
    c = 1 if len(img.shape) == 2 else img.shape[2]
    if dst is None:
        if c == 1:
            dst = np.zeros((w, h), np.uint8)
        else:
            dst = np.zeros((w, h, c), np.uint8)
    for i in range(w):
        for j in range(h):
            if c == 1:
                dst[i, j] = 255 - img[i, j]
            else:
                for k in range(c):
                    dst[i, j, k] = 255 - img[i, j, k]
    return img


def transparence2white(img: np.ndarray):
    '''
    透明背景会被当成黑色处理，有的时候白色更合适
    '''
    if len(img.shape) < 3 or img.shape[2] != 4:
        return img

    w, h = img.shape[0], img.shape[1]
    # dst = np.zeros((w, h, 3), np.uint8)
    for i in range(w):
        for j in range(h):
            if img[i][j][3] == 0:
                img[i, j] = [255, 255, 255, 255]
            # else:
            #     dst[i, j] = img[i, j][0:3]
    return img


def cv_imread_CN(image_path):
    '''
    读取中文路径下的图片
    支持透明通道
    '''
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return img


def cv_imwrite_CN(save_path, img):
    if os.path.exists(save_path):
        os.remove(save_path)
    tail = os.path.splitext(save_path)[1]
    cv2.imencode(tail, img)[1].tofile(save_path)


def black2Alpha(img):
    w, h = img.shape[0], img.shape[1]
    dst = np.zeros((w, h, 4), np.uint8)
    # 用黑色代表透明颜色的程度
    for i in range(w):
        for j in range(h):
            if len(img.shape) == 2:
                x = img[i, j]
                dst[i, j] = [x, x, x, x]
            else:
                temp = img[i][j]
                dst[i][j] = [temp[0], temp[1], temp[2], int(temp[0] * 0.75)]
    return dst


if __name__ == '__main__':
    src = Image.open(
        r"C:\Users\liugu\Documents\Tencent Files\2583657917\FileRecv\MobileFile\52337457625514808391616091655396.jpg")  # type:Image

    plt.figure("dog")
    plt.imshow(src)
    plt.show()

    src = np.array(src)
    res = black2Alpha(src)
    res = Image.fromarray(res)

    plt.figure('res')
    plt.imshow(res)
    plt.show()

    res.save(r"C:\Users\liugu\Documents\Tencent Files\2583657917\FileRecv\MobileFile\res.png")
