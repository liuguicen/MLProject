import cv2 as cv
import torch
print(torch.cuda.is_available())
print(cv.__version__)
from paddle import utils
utils.run_check()

import tensorflow as tf
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
import logging
logging.debug('111')