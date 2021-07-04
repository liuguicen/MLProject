"""找用于风格迁移的好"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
from PIL import Image
import tensorflow as tf

import FileUtil
import ImageUtil
import RunRecord


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


predictor = tf.lite.Interpreter(
    model_path=r'D:\MLProject\GAN\StyleTransfer\google_mogenta_transfer\style_predict_f16_256.tflite',
    num_threads=8)  # type:tf.lite.Interpreter
predictor.allocate_tensors()

transfer_interpreter = tf.lite.Interpreter(model_path="style_transfer_f16_384.tflite")
# 分配空间
transfer_interpreter.allocate_tensors()


def stylePredict(style_image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    style_image = style_image.resize((width, height))
    # add N dim
    input_data = np.expand_dims(style_image, axis=0)
    if len(input_data.shape) == 3:
        return None
    if input_data.shape[-1] == 4:
        input_data = input_data[:, :, :, 1:4]
    if input_data.shape[-1] == 2:
        return None
    if floating_model:
        input_data = (np.float32(input_data) - 0) / 255.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def transfer(content_image, style_predict, interpreter):
    if style_predict is None:
        return None
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    style_image = content_image.resize((width, height))
    # add N dim
    input_data = np.expand_dims(style_image, axis=0)
    if len(input_data.shape) == 3:
        return input_data
    if input_data.shape[-1] == 4:
        input_data = input_data[:, :, :, 1:4]
    if input_data.shape[-1] == 2:
        return input_data
    if floating_model:
        input_data = (np.float32(input_data) - 0) / 255.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.set_tensor(input_details[1]['index'], style_predict)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def styleTransfer(style, content):
    style_predict = stylePredict(style, predictor)
    res = transfer(content, style_predict, transfer_interpreter)  # type:np.ndarray
    if res is None:
        return None
    res = res.squeeze() * 255
    res = res.astype(np.uint8)
    img_rst = Image.fromarray(res)
    return img_rst


from matplotlib import pyplot as plt
import matplotlib
import run_record
import common_dataset
from os import path

def run(runRecord):
    styleList = FileUtil.getChildPath_firstLeve(
        path.join(common_dataset.dataset_dir, r'wikiart\StyleRstGoogleModel\选择的\resize_style_256'))
    contentList = FileUtil.getChildPath_firstLeve(
        path.join(common_dataset.dataset_dir, r'wikiart\StyleRstGoogleModel\可以做例子的\content'))
    rstDir = path.join(common_dataset.dataset_dir, r'wikiart\StyleRstGoogleModel\可以做例子的\rst')
    FileUtil.mkdir(rstDir)

    for id, stylePath in enumerate(styleList):
        styleName = 'style_' + os.path.splitext(os.path.basename(stylePath))[0]
        style = Image.open(stylePath)
        if id < runRecord.common_iter_count:
            continue
        if np.ndim(style) != 3:
            continue

        if stylePath.endswith('.png'):
            print(id, 'png skip')
            continue
        style.save(os.path.join(rstDir, styleName + '.jpg'))
        # if len(style.size) != 3:
        #     print(id, 'channel != 3 skip')
        #     continue
        for i, contentPath in enumerate(contentList):
            contentName = os.path.splitext(os.path.basename(contentPath))[0]
            content = Image.open(contentPath)
            content.save(os.path.join(rstDir, styleName + '_' + contentName + '.jpg'))
            rst = styleTransfer(style, content)
            if rst == None:
                rst = content
                print(i, '图像错误，')
            rst.save(os.path.join(rstDir, styleName + '_' + contentName + '_rst.jpg'))
            print(id, i, '保存完成')
        runRecord.common_iter_count = id
        runRecord.saveRunRecord()
        print(id, '  ', stylePath, 'save finish')


if __name__ == '__main__':
    finish = False
    runRecord = BaseRunRecord.readFromDisk()
    if runRecord is None:
        runRecord = BaseRunRecord.RunRecord()
    runRecord.common_iter_count += 1
    while not finish:
        try:
            run(runRecord)
            finish = True
        except Exception as e:
            print(e)
            runRecord.common_iter_count += 1
            pass

# top_k = results.argsort()[-5:][::-1]
# labels = load_labels(args.label_file)
# for i in top_k:
#     if floating_model:
#         print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
#     else:
#         print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
#
# print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
