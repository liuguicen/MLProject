import os

import cv2
import sys

import ImageUtil

path = r'E:\读研相关\过去的\建模\2019题\2019年中国研究生数学建模竞赛C题\2019年中国研究生数学建模竞赛C题\附件\车辆.mp4'
vc = cv2.VideoCapture(path)

ret_val = None
if vc.isOpened():
    ret_val = True
    print("Open video succeed...")
else:
    ret_val = False
    print("Open video failed...")

timeF = 200
c = 1
while ret_val:
    ret_val, frame = vc.read()
    # if c % timeF == 0:
    ImageUtil.saveImageNdArray(os.path.dirname(path) + '\\' + str(c) + '.jpg', frame)
    c = c + 1
vc.release()
