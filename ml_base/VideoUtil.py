import os
import random

import cv2
import subprocess

import ffmpeg
import numpy

import FileUtil

file_path = r'E:\重要_dataset_model\动画漫画\宫崎骏\天空之城.mp4'

import ImageUtil

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    print('ffmpeg find key frame')
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=', '').split()
    print('ffmpeg find key frame finish')
    return zip(range(len(frame_types)), frame_types)


from PIL import Image

from matplotlib import pyplot as plt


def extra_and_save_keyframes(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    key_frame_path = video_name + "key_frame_ids.txt"
    if not os.path.exists(key_frame_path):
        frame_types = get_frame_types(video_path)
        key_frame_id_list = [x[0] for x in frame_types if x[1] == 'I']
        FileUtil.writeList(key_frame_id_list, video_name + "key_frame_ids.txt")
    else:
        key_frame_id_list = FileUtil.readList(key_frame_path)

    print('frame number', len(key_frame_id_list))
    if key_frame_id_list:
        res_dir = os.path.join(os.path.dirname(video_path), video_name + '_key_frame')
        FileUtil.mkdir(res_dir)
        cap = cv2.VideoCapture(video_path)
        for i, frame_no in enumerate(key_frame_id_list):
            frame_no = int(frame_no)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
            ret, frame = cap.read()
            outname = video_name + '_' + str(i) + '_frame_' + str(frame_no) + '.jpg'
            # cv2.imshow('video', frame)
            ImageUtil.cv_save_image_CN(os.path.join(res_dir, outname), frame)
            print('Saved: ' + outname)
        print('save key frame finish')
        cap.release()
    else:
        print('No I-frames in ' + video_path)


if __name__ == '__main__':
    extra_and_save_keyframes(file_path)
