import torch

from common_lib_import_and_set import *
from functools import reduce

def caculate_cos(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    '''
    计算张量的余弦相似度
    任意维度的张量
    '''
    a = tensor_a.view(-1).cpu()
    b = tensor_b.view(-1).cpu()
    # 先拉平，好计算
    # 多维向量夹角计算公式 cos(t) = a 点积 b / (|a||b|)
    ab = 0
    for id, x in enumerate(a):
        ab += a[id] * b[id]
    a_m = math.sqrt(reduce(lambda s, x: s + x ** 2, a, 0))
    b_m = math.sqrt(reduce(lambda s, x: s + x ** 2, b, 0))
    theta = math.acos(ab / (a_m * b_m)) / math.pi * 180
    return theta

if __name__ == "__main__":
    caculate_cos(torch.tensor([1,1]), torch.tensor([1, 3]))