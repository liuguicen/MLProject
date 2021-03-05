import cv2
import numpy as np

def cv_imread_CN(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return img


def cv_imwrite_CN(save_path, img):
    cv2.imencode('.jpg', img)[1].tofile(save_path)
