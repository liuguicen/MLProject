import numpy as np
import cv2


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


def float2int(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    np.array([img], dtype='uint8')
    return img