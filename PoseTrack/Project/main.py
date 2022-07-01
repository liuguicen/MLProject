import argparse
import os
import os.path as osp
import time
import cv2
import numpy as np
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from common_lib_import_and_set import *
from datetime import datetime

from ByteTrack.ByteTrack.yolox.tracker.kalman_filter import KalmanFilter
from ByteTrack.ByteTrack.yolox.tracker.byte_tracker import STrack
from LightTrackV2.keypoint_paddle.one_human_kp_detector import OneHumanKpDetector_Paddle

from ml_base.detect.detector_nanodet import NanoHumanDetector


class PoseTrackConfig:
    min_box_area = 10
    aspect_ratio_thresh = 1.6
    video_path = "/E/dataset/ObjectTracking/lasot/person/person-3/video.mp4"
    output_dir = "/D/MLProject/PoseTrack/Project/output"
    save_result = True
    device = "gpu"
    enlarge_scale = 0.2  # how much to enlarge the bbox before pose estimation


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def drawKp(bk_img, keypoints, kp1 = None):
    for i, kp in enumerate(keypoints):
        # 在图片上添加文字信息
        cv2.putText(bk_img, str(i), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 1, cv2.LINE_AA)
    # x = kp1[0::3]
    # y = kp1[1::3]
    # for i in range(len(x)):
    #     # 在图片上添加文字信息
    #     cv2.putText(bk_img, str(i), (int(x[i]), int(y[i])), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    # # 显示图片
    # cv2.imshow("add_text", bk_img)
    # cv2.waitKey()

def enlarge_bbox(bbox, scale):
    '''
    x1,y1,x2,y2
    '''
    assert (scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged

def get_bbox_from_keypoints(keypoints_python_data):
    '''
    keypoints_python_data [x0,y0,score0,x1,y1,score1]
    '''
    if keypoints_python_data == [] or keypoints_python_data == 45 * [0]:
        return [0, 0, 2, 2]

    x_list = []
    y_list = []
    for kp in keypoints_python_data:
        x = kp[0]
        y = kp[1]
        vis = kp[2]
        if vis != 0 and vis != 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    scale = 0.001  # 不放大
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    bbox_in_xywh = STrack.x2y2_to_wh(bbox)
    return bbox_in_xywh


kalman_filter = KalmanFilter()


def video_demo(human_detector: NanoHumanDetector, pose_dtector: OneHumanKpDetector_Paddle, output_dir, current_time):
    config = PoseTrackConfig
    cap = cv2.VideoCapture(config.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = osp.join(output_dir, config.video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")

    detectTimer = Timer()
    traceTimer = Timer()
    frame_id = 0
    results = []
    frame_path_list = []
    # 整个流程:
    # 进入视频, 检测视频中的人体框，然后选择一个人体框，
    # 利用人体框初始化一条轨迹 continue下一帧
    # 利用轨迹预测的当前帧的box，检测关键点，根据关键点获取人体框，
    # 根据关键点和人体框，判断关联程度，然后判断是否目标丢失，
    # 如果目标没有丢失，那么就将它添加到轨迹里面，更新轨迹数据

    # 如果丢失了，那么在更大的范围执行人体检测，然后对每个检测到的目标执行关键点检测并判断关联程度，
    # 然后判断是否所有目标都不匹配，然后判断丢失，这一步也可以先不做，直接判断目标丢失

    # 如果连续多帧都丢失了，则判断跟踪完成了
    #
    #
    # 轨迹就3步，初始化，预测，更新
    strack = None
    while True:
        frame_id += 1
        if frame_id % 20 == 0:
            logger.info('detect frame {} ({:.2f} fps)'
                        .format(frame_id, 1. / max(1e-5, detectTimer.average_time)))
            logger.info('track frame {} ({:.2f} fps)'
                        .format(frame_id, 1. / max(1e-5, traceTimer.average_time)))
            detectTimer.clear()
            traceTimer.clear()
        ret_val, frame = cap.read()
        if not ret_val:
            break
        detectTimer.tic()
        outputs = human_detector.infer(frame)
        detectTimer.toc()
        # 选择第一个人体框
        if outputs[0] is None:
            logger.info(f'frame {frame_id} no human')
            continue
            # 初始化轨迹和滤波器
        if strack is None:
            human_box = outputs[0]
            strack = STrack(human_box[2:], human_box[1])
            strack.activate(kalman_filter, frame_id)

        # 获取预测框，扩大，然后进行姿态检测
        kp_input = []
        kp_input.append([0, strack.score])
        tlbr = enlarge_bbox(strack.tlbr, 0.2)
        kp_input[0].extend(tlbr)
        kp = pose_dtector.detectPeopleKp(frame, {"boxes": np.array(kp_input), "boxes_num": np.array([1])}, det_threshold=0.2)
        kp_list = kp['keypoint'][0][0]
        score = kp['keypoint'][1][0][0]
        # 判断目标是否丢失
        if score > 0.2:
            tlwh = get_bbox_from_keypoints(kp_list)
            strack_cur_frame = STrack(tlwh, score)
            strack.update(strack_cur_frame, frame_id)
        else:
            logger.info(f"{frame_id} 关键点可信度低，判定目标丢失")
            pass
            # strack.re_activate()

        online_tlwhs = []
        online_ids = []
        online_scores = []
        vertical = strack.tlwh[2] / strack.tlwh[3] > config.aspect_ratio_thresh
        if strack.tlwh[2] * strack.tlwh[3] > config.min_box_area and not vertical:
            online_tlwhs.append(strack.tlwh)
            online_ids.append(0)
            online_scores.append(strack.score)
            results.append(
                f"{frame_id},{strack.tlwh[0]:.2f},{strack.tlwh[1]:.2f},{strack.tlwh[2]:.2f},{strack.tlwh[3]:.2f},{strack.score:.2f},-1,-1,-1\n"
            )
        traceTimer.toc()
        drawKp(frame, kp_list)
        online_im = plot_tracking(
            frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / traceTimer.average_time
        )


        if config.save_result:
            framePath = path.join(output_dir, "frameList", str(frame_id) + ".jpg")
            image_util.saveImageNdArray(framePath, online_im)
            frame_path_list.append(framePath)
            # vid_writer.write(online_im)

    video_util.make_video_from_images(frame_path_list, save_path, fps=fps, size=(int(width), int(height)))

    if config.save_result:
        res_file = osp.join(output_dir, f"{config.experiment_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


if __name__ == "__main__":
    output_dir = PoseTrackConfig.output_dir
    os.makedirs(output_dir, exist_ok=True)

    video_demo(NanoHumanDetector(), OneHumanKpDetector_Paddle(), output_dir, time.time())
