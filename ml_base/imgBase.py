import os

import cv2
import numpy as np


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
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return img


if __name__ == '__main__':
    img = transparence2white(cv_imread_CN('111.png'))
    img_w, img_h = img.shape[0], img.shape[1]
    ratio = max(128 / img_w, 128 / img_h)
    img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('..', img)
    cv2.waitKey(0)


def cv_imwrite_CN(save_path, img):
    if os.path.exists(save_path):
        os.remove(save_path)
    cv2.imencode('.jpg', img)[1].tofile(save_path)
