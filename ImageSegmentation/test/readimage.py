import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def readNormalImg(path):
    # 图像归一化
    transform_GY = transforms.ToTensor()  # 将PIL.Image转化为tensor，即归一化。
    # 注：shape 会从(H，W，C)变成(C，H，W)

    # 图像规范化
    transform_BZ = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # 取决于数据集
        std=[0.5, 0.5, 0.5]
    )

    # transform_compose
    transform_compose = transforms.Compose([
        # 先归一化再规范化
        transform_GY,
        transform_BZ
    ])

    img = Image.open(path)
    # (H, W, C)变为(C, H, W)
    img_transform = transform_compose(img)
    return img_transform
