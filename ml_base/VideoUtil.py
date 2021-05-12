import os
import random

import cv2
import subprocess

import ffmpeg
import numpy

import FileUtil

file_path = r'G:\迅雷下载\[阳光电影www.ygdy8.com].起风了.BD.720p.国粤日三语中字.mkv'
import ImageBase

# file_path = r'E:\读研相关\过去的\建模\2019题\2019年中国研究生数学建模竞赛C题\2019年中国研究生数学建模竞赛C题\附件\车辆.mp4'


def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    print('start decode')
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=', '').split()
    return zip(range(len(frame_types)), frame_types)


from PIL import Image


def save_i_keyframes(video_fn):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    print('frame number', len(i_frames))
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        res_dir = os.path.join(os.path.dirname(video_fn), basename + '_frame')
        FileUtil.mkdir(res_dir)
        cap = cv2.VideoCapture(video_fn)
        for i, frame_no in enumerate(i_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = basename + '_' + str(i) + '_frame_' + str(frame_no) + '.jpg'
            Image.fromarray(frame).save(os.path.join(res_dir, outname))
            print('Saved: ' + outname)
        cap.release()
    else:
        print('No I-frames in ' + video_fn)


if __name__ == '__main__':
    save_i_keyframes(file_path)


def read_frame_as_jpeg(in_file, frame_num):
    """
    指定帧数读取任意帧
    """
    out, err = (
        ffmpeg.input(in_file)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
            .run(capture_stdout=True)
    )
    return out


def get_video_info(in_file):
    """
    获取视频基本信息
    """
    try:
        probe = ffmpeg.probe(in_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        return video_stream
    except ffmpeg.Error as err:
        print(str(err.stderr, encoding='utf8'))


# if __name__ == '__main__':
#     video_info = get_video_info(file_path)
#     total_frames = int(video_info['nb_frames'])
#     print('总帧数：' + str(total_frames))
#     random_frame = random.randint(1, total_frames)
#     for i in range(total_frames):
#         print('随机帧：' + str(random_frame))
#         out = read_frame_as_jpeg(file_path, i)
#         image_array = numpy.asarray(bytearray(out), dtype="uint8")
#         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         ImageBase.cv_imwrite_CN(str(i), image)
