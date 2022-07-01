# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import cv2
import math
import numpy as np
import paddle
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

from det_keypoint_unite_utils import argsparser
from preprocess import decode_image
from infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images
from keypoint_infer import KeyPoint_Detector, PredictConfig_KeyPoint
from visualize import draw_pose
from benchmark_utils import PaddleInferBenchmark
from keypoint_paddle.utils import get_current_memory_mb
from keypoint_postprocess import translate_to_ori_images

KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def bench_log(detector, img_list, model_info, batch_size=1, name=None):
    mems = {
        'cpu_rss_mb': detector.cpu_mem / len(img_list),
        'gpu_rss_mb': detector.gpu_mem / len(img_list),
        'gpu_util': detector.gpu_util * 100 / len(img_list)
    }
    perf_info = detector.det_times.report(average=True)
    data_info = {
        'batch_size': batch_size,
        'shape': "dynamic_shape",
        'data_num': perf_info['img_num']
    }

    log = PaddleInferBenchmark(detector.config, model_info, data_info,
                               perf_info, mems)
    log(name)


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, det_threshold,
                           keypoint_threshold, run_benchmark):
    '''
    det_res 人体框的 x,y,w,h
    '''
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res, det_threshold)
    keypoint_vector = []
    score_vector = []
    rect_vector = det_rects
    batch_loop_cnt = math.ceil(float(len(rec_images)) / keypoint_batch_size)

    for i in range(batch_loop_cnt):
        start_index = i * keypoint_batch_size
        end_index = min((i + 1) * keypoint_batch_size, len(rec_images))
        batch_images = rec_images[start_index:end_index]
        batch_records = np.array(records[start_index:end_index])
        if run_benchmark:
            # warmup
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, repeats=10, add_timer=False)
            # run benchmark
            keypoint_result = keypoint_detector.predict(
                batch_images, keypoint_threshold, repeats=10, add_timer=True)
        else:
            keypoint_result = keypoint_detector.predict(batch_images,
                                                        keypoint_threshold)
        orgkeypoints, scores = translate_to_ori_images(keypoint_result,
                                                       batch_records)
        keypoint_vector.append(orgkeypoints)
        score_vector.append(scores)

    keypoint_res = {}
    keypoint_res['keypoint'] = [
        np.vstack(keypoint_vector).tolist(), np.vstack(score_vector).tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          image_list,
                          keypoint_batch_size=1,
                          save_res=False):
    det_timer = detector.get_timer()
    store_res = []
    for i, img_file in enumerate(image_list):
        # Decode image in advance in det + pose prediction
        det_timer.preprocess_time_s.start()
        image, _ = decode_image(img_file, {})
        det_timer.preprocess_time_s.end()

        if FLAGS.run_benchmark:
            # warmup
            results = detector.predict(
                [image], FLAGS.det_threshold, repeats=10, add_timer=False)
            # run benchmark
            results = detector.predict(
                [image], FLAGS.det_threshold, repeats=10, add_timer=True)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
        else:
            logging.error('human start')
            results = detector.predict([image], FLAGS.det_threshold)
            logging.error('human end')


        if results['boxes_num'] == 0:
            continue

        logging.error('kp start')
        keypoint_res = predict_with_given_det(
            image, results, topdown_keypoint_detector, keypoint_batch_size,
            FLAGS.det_threshold, FLAGS.keypoint_threshold, FLAGS.run_benchmark)
        logging.error('kp end')

        if save_res:
            store_res.append([
                i, keypoint_res['bbox'],
                [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
            ])
        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            topdown_keypoint_detector.cpu_mem += cm
            topdown_keypoint_detector.gpu_mem += gm
            topdown_keypoint_detector.gpu_util += gu
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            draw_pose(
                img_file,
                keypoint_res,
                visual_thread=FLAGS.keypoint_threshold,
                save_dir=FLAGS.output_dir)
    if save_res:
        """
        1) store_res: a list of image_data
        2) image_data: [imageid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_image_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def topdown_unite_predict_video(detector,
                                topdown_keypoint_detector,
                                camera_id,
                                keypoint_batch_size=1,
                                save_res=False):
    video_name = 'output.mp4'
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.splitext(os.path.basename(FLAGS.video_file))[
                         0] + '.mp4'
    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 0
    store_res = []
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        index += 1
        print('detect frame: %d' % (index))

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.predict([frame2], FLAGS.det_threshold)

        keypoint_res = predict_with_given_det(
            frame2, results, topdown_keypoint_detector, keypoint_batch_size,
            FLAGS.det_threshold, FLAGS.keypoint_threshold, FLAGS.run_benchmark)

        im = draw_pose(
            frame,
            keypoint_res,
            visual_thread=FLAGS.keypoint_threshold,
            returnimg=True)
        if save_res:
            store_res.append([
                index, keypoint_res['bbox'],
                [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
            ])

        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    writer.release()
    if save_res:
        """
        1) store_res: a list of frame_data
        2) frame_data: [frameid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_video_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def main():
    pred_config = PredictConfig(FLAGS.det_model_dir)
    detector_func = 'Detector'
    if pred_config.arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'
    # 人体目标检测器
    detector = eval(detector_func)(pred_config,
                                   FLAGS.det_model_dir,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig_KeyPoint(FLAGS.keypoint_model_dir)
    assert KEYPOINT_SUPPORT_MODELS[
               pred_config.
                   arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'
    topdown_keypoint_detector = KeyPoint_Detector(
        pred_config,
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.keypoint_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                    FLAGS.camera_id, FLAGS.keypoint_batch_size,
                                    FLAGS.save_res)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(detector, topdown_keypoint_detector, img_list,
                              FLAGS.keypoint_batch_size, FLAGS.save_res)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            topdown_keypoint_detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.det_model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')
            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                      FLAGS.keypoint_batch_size, 'KeyPoint')





class OneHumanKpDetector_Paddle:

    def __init__(self):
        self.loadModel()

    def loadModel(self):
        # paddle.enable_static()
        parser = argsparser()
        self.FLAGS = parse_arg(parser)
        print_arguments(self.FLAGS)
        self.FLAGS.device = self.FLAGS.device.upper()
        assert self.FLAGS.device in ['CPU', 'GPU', 'XPU'
                                     ], "device should be CPU, GPU or XPU"
        pred_config = PredictConfig_KeyPoint(self.FLAGS.keypoint_model_dir)
        assert KEYPOINT_SUPPORT_MODELS[
                   pred_config.
                       arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'

        self.kp_detector = KeyPoint_Detector(
            pred_config,
            self.FLAGS.keypoint_model_dir,
            device=self.FLAGS.device,
            run_mode=self.FLAGS.run_mode,
            batch_size=self.FLAGS.keypoint_batch_size,
            trt_min_shape=self.FLAGS.trt_min_shape,
            trt_max_shape=self.FLAGS.trt_max_shape,
            trt_opt_shape=self.FLAGS.trt_opt_shape,
            trt_calib_mode=self.FLAGS.trt_calib_mode,
            cpu_threads=self.FLAGS.cpu_threads,
            enable_mkldnn=self.FLAGS.enable_mkldnn,
            use_dark=self.FLAGS.use_dark)

    def detectPeopleKp(self, img, box_results, det_threshold = 0.5):
        '''
        检测一个人体的关键点，
        img 完整图片
        box_results 人体框检测结果
        结构 = dict{
        'boxes':  2维ndarray = { 6维人体框数据的列表=种类，分数，box(x,y,w,h), ... }
        'boxes_num':一维ndarray，人体框个数

        返回值:
dict {
'keypoint':[所有人体关键点数据列表[套了一层啥都没有[关键点的列表[关键点数据列表[x,y, score]]],
            所有人体关键点得分列表[套了一层啥都没有[得分]]]
'bbox':[所有人体框列表[人体框列表]]
}
        '''
        # predict from image
        # img_list = get_test_images(self.FLAGS.image_dir, self.FLAGS.image_file)

        keypoint_res = predict_with_given_det(
            img, box_results, self.kp_detector, self.FLAGS.keypoint_batch_size,
            det_threshold, self.FLAGS.keypoint_threshold, self.FLAGS.run_benchmark)

        # mode = self.FLAGS.run_mode
        # keypoint_model_dir = self.FLAGS.keypoint_model_dir
        # keypoint_model_info = {
        #     'model_name': keypoint_model_dir.strip('/').split('/')[-1],
        #     'precision': mode.split('_')[-1]
        # }
        # bench_log(self.kp_detector, img, keypoint_model_info,
        #           self.FLAGS.keypoint_batch_size, 'KeyPoint')
        return keypoint_res


# 整个关键点检测的流程是，创建人体检测器，创建关键点检测器，用人体检测器检测出所有的人体框，
# 从整个图片中根据人体框分割出人体图片，人体框图片输入关键点检测器，得到关键点，关键点数据结构，x，y，score，一共17个关键点
def parse_arg(parser):
    FLAGS = parser.parse_args()
    FLAGS.det_model_dir = '/D/MLProject/PoseTrack/LightTrackV2/weights/pp-predestrain/picodet_s_320_pedestrian'
    FLAGS.keypoint_model_dir = '/D/MLProject/PoseTrack/LightTrackV2/weights/pp-tinypose/tinypose_256x192'
    FLAGS.device = 'GPU'
    return FLAGS


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parse_arg(parser)
    FLAGS.det_threshold = 0.35
    FLAGS.image_dir = None
    # FLAGS.image_file = "/D/MLProject/PoseTrack/LightTrackV2/data/demo/video/frame00000.jpg"
    FLAGS.image_file = '/D/MLProject/PoseTrack/LightTrackV2/data/demo/video_out_img/video_test/frame00047.jpg'
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    FLAGS.save_res = True
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
