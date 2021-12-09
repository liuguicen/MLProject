import imageio
from PIL import Image
import numpy as np


def load_image(filename, size, crop):
    image = imageio.imread(filename)
    if crop:
        image = central_crop(image)
    if size:
        image = scale_image(image, size)
    return image


def prepare_image(image, normalize=True, data_format='channels_first'):
    if normalize:
        image = image.astype(np.float32)
        image /= 255
    if data_format == 'channels_first':
        image = np.transpose(image, [2, 0, 1])  # HWC --> CHW
    return image


def scale_image(image, size):
    "size specifies the minimum height or width of the output"
    h, w, _ = image.shape
    if h > w:
        image = np.array(Image.fromarray(image).resize((h * size // w, size), resample=Image.BILINEAR))
    else:
        image = np.array(Image.fromarray(image).resize((size, w * size // h), resample=Image.BILINEAR))
    return image


def central_crop(image):
    h, w, _ = image.shape
    minsize = min(h, w)
    h_pad, w_pad = (h - minsize) // 2, (w - minsize) // 2
    image = image[h_pad:h_pad + minsize, w_pad:w_pad + minsize]
    return image


def save_image(filename, image, data_format='channels_first'):
    if data_format == 'channels_first':
        image = np.transpose(image, [1, 2, 0])  # CHW --> HWC
    image *= 255
    image = np.clip(image, 0, 255)
    Image.fromarray(image).save(filename, image.astype(np.uint8))


def load_mask(filename, h, w):
    mask = imageio.imread(filename, mode='L')
    mask = np.array(Image.fromarray(mask).resize((h, w), resample=Image.BILINEAR))
    mask = mask.astype(np.uint8)
    mask[mask == 255] = 1
    return mask
