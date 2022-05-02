from common_lib_import_and_set import *


def make_video_from_images(img_paths, outvid_path, fps=25, size=None,
                           is_color=True, format="XVID"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for ct, img_path in enumerate(img_paths):
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        img = imread(img_path)
        if img is None:
            print(img_path)
            continue
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid_path, fourcc, float(fps), size, is_color)

        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    if vid is not None:
        vid.release()
    return vid
from common_lib_import_and_set import *

file_path = path.join(common_dataset.dataset_dir, r'动画漫画\宫崎骏\天空之城.mp4')


def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    # print('ffmpeg find key frame')
    # out = subprocess.check_output(command + [video_fn]).decode()
    # frame_types = out.replace('pict_type=', '').split()
    # print('ffmpeg find key frame finish')
    # return zip(range(len(frame_types)), frame_types)


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
            image_util.cv_save_image_CN(os.path.join(res_dir, outname), frame)
            print('Saved: ' + outname)
        print('save key frame finish')
        cap.release()
    else:
        print('No I-frames in ' + video_path)


def changeVideoSize(inPath, outPath):
    videoCapture = cv2.VideoCapture(inPath)

    fps = 30  # 保存视频的帧率,可改变
    size = (18, 32)  # 保存视频大小

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(outPath,
                                  fourcc, fps, size)

    while True:
        success, frame = videoCapture.read()
        if success:
            img = cv2.resize(frame, size)
            videoWriter.write(img)
        else:
            print('break')
            break

    # 释放对象，不然可能无法在外部打开
    videoWriter.release()


def createTestVideo(outPath):
    w = 9
    h = 16
    img = np.zeros((16, 9, 3), np.uint8)
    # start     x     y     end     x     y     color
    # # 红色
    # cv2.rectangle(img, (0 * w // 3, 0), (1 * w // 3, h // 2), (0, 0, 255), thickness=-1)
    # # 绿色
    # cv2.rectangle(img, (1 * w // 3, 0), (2 * w // 3, h // 2), (0, 255, 0), thickness=-1)
    # # 蓝色
    # cv2.rectangle(img, (2 * w // 3, 0), (3 * w // 3, h // 2), (255, 0, 0), thickness=-1)
    #
    # # 灰色
    # cv2.rectangle(img, (0 * w // 3, 1 * h // 2), (1 * w // 3, h), (128, 128, 128), thickness=-1)
    # # 红绿
    # cv2.rectangle(img, (1 * w // 3, 1 * h // 2), (2 * w // 3, h), (0, 255, 255), thickness=-1)
    # # RB
    # cv2.rectangle(img, (2 * w // 3, 1 * h // 2), (3 * w // 3, h), (255, 0, 255), thickness=-1)

    cv2.rectangle(img, (0 * w // 3, 0), (3 * w // 3, 2 * h // 2), (255, 255, 0), thickness=-1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(outPath,
                                  fourcc, 30, (w, h))

    for i in range(60):
        videoWriter.write(img)

    # 释放对象，不然可能无法在外部打开
    videoWriter.release()


if __name__ == '__main__':
    # changeVideoSize("/D/MLProject/ml_base/VideoUtil/test_video2.mp4",
    #                 "/D/MLProject/ml_base/VideoUtil/test_video3.mp4")
    createTestVideo("/D/MLProject/ml_base/VideoUtil/test_videobg.mp4")
