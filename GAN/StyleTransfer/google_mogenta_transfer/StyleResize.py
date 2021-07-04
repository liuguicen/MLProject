import os
import re
import traceback

import PIL.Image as Image
import numpy as np
import common_dataset
from os import path

name = r'style_100398_ss.jpg'
print(re.search(r'\d_', name))

root = path.join(common_dataset.dataset_dir, r'wikiart\StyleRstGoogleModel\选择的')
allList = os.listdir(root)
for name in allList:
    path = os.path.join(root, name)
    if (not os.path.isdir(path)) and re.search(r'\d_', name) is None:
        try:
            print('开始resize ', path)
            image = Image.open(path)  # type:Image
            H, W = image.width, image.height
            if H < W:
                ratio = W / H
                H = 256
                W = int(ratio * H)
            else:
                ratio = H / W
                W = 256
                H = int(ratio * W)
            image = image.resize((H, W))
            name = re.findall(r'\d+', path)[0]
            image.save(os.path.join(root + r"\resize_style_256", name + "_resize_256.jpg"))
        except Exception as e:
            traceback.print_exc()
            print(e)
            print('出错了 ', path)
            continue
