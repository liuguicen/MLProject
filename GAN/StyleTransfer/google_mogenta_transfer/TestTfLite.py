import os
import cv2
import numpy as np
import tensorflow as tf
import glob
from PIL import Image
import time

import common_dataset
from os import path

# *****************************************测试图片、模型路径******************************************
# 图片集文件夹名称
testFileName = '1'
# 图片集路径
image_dir = path.join(common_dataset.dataset_dir, r'wikiart\train')
# 得到图片集列表
img_name_list = glob.glob(image_dir + os.sep + '*')

predictor = tf.lite.Interpreter(model_path="style_predict_f16_256.tflite")
# 分配空间
predictor.allocate_tensors()

transfer = tf.lite.Interpreter(model_path="style_transfer_f16_384.tflite")
# 分配空间
transfer.allocate_tensors()

# 获取输入与输出的详细信息
input_details = predictor.get_input_details()
output_details = predictor.get_output_details()
print("input_details =", input_details)
print("output_details =", output_details)

input_details = transfer.get_input_details()
output_details = transfer.get_output_details()
print("input_details =", input_details)
print("output_details =", output_details)


# ***********************************************归一化化输入*****************************************
def format_input(input_image):
    inputs = np.array(input_image)
    if inputs.shape[-1] == 4:
        input_image = input_image.convert('RGB')
    return np.expand_dims(np.array(input_image) / 255., 0)


DEFAULT_STYLE_SHAPE = [255, 255, 3]


# ***********************************************主函数**********************************************
def main():
    # 循环加载待检测图片
    for img in img_name_list:
        print("img =", img)
        start_time = time.time()
        # 将图片转化为RGB格式
        image = Image.open(img).convert('RGB')
        input_image = image

        # 将输入图片缩放到模型的输入尺寸大小
        if image.size != DEFAULT_STYLE_SHAPE:
            input_image = image.resize(DEFAULT_STYLE_SHAPE[:2], Image.BICUBIC)

        # 输入图片归一化处理并转化为需要的格式
        input_tensor = format_input(input_image)
        input_tensor = input_tensor.astype(np.float32)

        # 为分配的张量赋值
        index = input_details[0]['index']
        transfer.set_tensor(index, input_tensor)

        # 调用解释器
        transfer.invoke()
        # 获得输出
        print("output_details[0]['index'] =", transfer.get_tensor(output_details[0]['index']))
        output_data = transfer.get_tensor(output_details[0]['index'])[0][0]
        # 去除不需要的维度，并将数据转化为数组形式
        result = np.squeeze(output_data)
        output_mask = np.asarray(result)

        # 缩放到原图大小
        if image.size != DEFAULT_IN_SHAPE:
            output_mask = cv2.resize(output_mask, dsize=image.size)

        # 转化为3通道灰度图
        output_image = cv2.cvtColor(output_mask.astype('float32'), cv2.COLOR_BGR2RGB) * 255.
        print("host_time =", time.time() - start_time)
        # 转化为单通道灰度图
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
        output_location = output_dir.joinpath(pathlib.Path(img).name)
        cv2.imwrite(str(output_location), output_image)


if __name__ == '__main__':
    main()
