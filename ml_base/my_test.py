import math
from functools import *

import torch

import image_util
from common_lib_import_and_set import *


def caculate_cos(last_backbone_feature: torch.Tensor, backbone_feature: torch.Tensor):
    a = last_backbone_feature.view(-1)
    b = backbone_feature.view(-1)
    ab = 0
    for id, x in enumerate(a):
        ab += a[id] * b[id]

    a_m = math.sqrt(reduce(lambda s, x: s + x ** 2, a, 0))
    b_m = math.sqrt(reduce(lambda s, x: s + x ** 2, b, 0))
    theta = math.acos(ab / (a_m * b_m)) / math.pi * 180
    return theta


if __name__ == "__main__":
    # print(reduce(lambda x, y: x + y ** 2, [1, 2, 3, 4, 5]))  # 使用 lambda 匿名函数

    videoCapture = cv2.VideoCapture()
    videoCapture.open('/D/MLProject/PoseTrack/LightTrackV2/demo/video1.mp4')

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    # fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
    print("fps=", fps, "frames=", frames)

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        image_util.showImage(frame)
        cv2.imwrite("frames(%d).jpg" % i, frame)
