# -*- coding: utf-8 -*-

import cv2
import numpy as np

from ml_base import fileUtil


def smoothEdge(edges):
    vesselImage = cv2.threshold(edges, 125, 255, cv2.THRESH_BINARY)
    blurredImage = cv2.pyrUp(vesselImage)

    for i in range(5):
        blurredImage = cv2.medianBlur(blurredImage, 7)

    cv2.pyrDown(blurredImage, blurredImage)
    cv2.threshold(blurredImage, blurredImage, 200, 255, cv2.THRESH_BINARY)


# img = fileUtil.cv_imread_CN('F:\\重要_data_set__big_size\\表情\\大黄脸\\2_02.png')
img = fileUtil.cv_imread_CN(r'D:\MLProject\common_data\表情\valid\熊猫头表情\熊猫头对讲机-一组熊猫头表情包原图.jpg')


def useCanny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


import time

import edge.Hed_edge_detect as hed
import imgBase

start = time.time()
edges = hed.detectEdge(img)
# edges = useCanny(img)
print('耗时 = ', time.time() - start)
# edges = smoothEdge(edges)

edges = np.uint8(edges * 255)
edges = imgBase.colorConvert(edges, edges)
cv2.imshow('edges', edges)
fileUtil.cv_imwrite_CN(r'edge.jpg', edges)
cv2.waitKey(0)
