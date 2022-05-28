import math
from functools import *

import torch

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
    print(caculate_cos(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 7])))
