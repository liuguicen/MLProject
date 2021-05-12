import glob
import logging
import os

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from ml_base import FileUtil
import Din_Config

# 代码完全源自 adain
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms=trans):
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'

        if not os.path.exists(content_dir_resized):
            os.mkdir(content_dir_resized)
        if not os.path.exists(style_dir_resized):
            os.mkdir(style_dir_resized)

        self._resize(content_dir, content_dir_resized)
        self._resize(style_dir, style_dir_resized)
        content_images = glob.glob((content_dir_resized + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir_resized + '/*')
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = transforms

    @staticmethod
    def _resize(source_dir, target_dir):

        print(f'Start resizing {source_dir} ')
        runRecord = Din_Config.runRecord
        start = runRecord.pre_process_content if 'COCO' in source_dir else runRecord.pre_process_style

        fileList = os.listdir(source_dir)
        if start >= len(fileList):
            return
        for i, item in tqdm(enumerate(fileList)):
            if i < start:
                continue
            filename = os.path.basename(item)
            try:
                image = io.imread(os.path.join(source_dir, item))
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(target_dir, filename), image)
                    if i % 100 == 0:
                        if 'COCO' in source_dir:
                            runRecord.pre_process_content = i
                        else:
                            runRecord.pre_process_style = i
                        FileUtil.saveRunRecord(runRecord, Din_Config.runRecordPath)
            except:
                logging.error('imge ', i, 'resize failed', source_dir)
                continue
        if 'COCO' in source_dir:
            runRecord.pre_process_content = len(fileList) + 1
        else:
            runRecord.pre_process_style = len(fileList) + 1
        FileUtil.saveRunRecord(runRecord, Din_Config.runRecordPath)
    
    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image, style_image = self.images_pairs[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        # content_image = io.imread(content_image, plugin='pil')
        # style_image = io.imread(style_image, plugin='pil')
        # Unfortunately,RandomCrop doesn't work with skimage.io
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return content_image, style_image
