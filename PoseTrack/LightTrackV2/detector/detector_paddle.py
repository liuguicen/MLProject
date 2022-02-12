# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings

warnings.filterwarnings('ignore')
import glob

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model

from ppdet.utils.logger import setup_logger

logger = setup_logger('train')


class PaddleHumanDetector:

    def __init__(self):
        self.loadModel()

    def parse_args(self):
        parser = ArgsParser()
        # parser.add_argument(
        #     "--config",
        #     type=str,
        #     default="/D/tools/PaddleDetection/configs/picodet/application/pedestrian_detection/picodet_s_320_pedestrian.yml")

        parser.add_argument(
            "--infer_dir",
            type=str,
            default=None,
            help="Directory for images to perform inference on.")
        parser.add_argument(
            "--infer_img",
            type=str,
            default=None,
            help="Image path, has higher priority over --infer_dir")
        parser.add_argument(
            "--output_dir",
            type=str,
            default="output",
            help="Directory for storing the output visualization files.")
        parser.add_argument(
            "--draw_threshold",
            type=float,
            default=0.5,
            help="Threshold to reserve the result for visualization.")
        parser.add_argument(
            "--slim_config",
            default=None,
            type=str,
            help="Configuration file of slim method.")
        parser.add_argument(
            "--use_vdl",
            type=bool,
            default=False,
            help="Whether to record the data to VisualDL.")
        parser.add_argument(
            '--vdl_log_dir',
            type=str,
            default="vdl_log_dir/image",
            help='VisualDL logging directory for image.')
        parser.add_argument(
            "--save_txt",
            type=bool,
            default=False,
            help="Whether to save inference result in txt.")
        args = parser.parse_args_for_lightTrack()
        return args

    def get_test_images(self, infer_dir, infer_img):
        """
        Get image path list in TEST mode
        """
        assert infer_img is not None or infer_dir is not None, \
            "--infer_img or --infer_dir should be set"
        assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
        assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

        # infer_img has a higher priority
        if infer_img and os.path.isfile(infer_img):
            return [infer_img]

        images = set()
        infer_dir = os.path.abspath(infer_dir)
        assert os.path.isdir(infer_dir), \
            "infer_dir {} is not a directory".format(infer_dir)
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
        images = list(images)

        assert len(images) > 0, "no image found in {}".format(infer_dir)
        logger.info("Found {} inference images in total.".format(len(images)))

        return images

    def infer(self, imgPath):
        '''
        返回框的形式是x，y,w,h
        '''

        # get inference images
        # images = self.get_test_images(None, imgPath)

        # inference
        return self.trainer.predict_for_human(
            [imgPath],
            draw_threshold=self.FLAGS.draw_threshold,
            output_dir=self.FLAGS.output_dir,
            save_txt=self.FLAGS.save_txt)

    def loadModel(self):
        self.FLAGS = self.parse_args()
        self.FLAGS.config = '/D/tools/PaddleDetection/configs/picodet/application/pedestrian_detection/picodet_s_320_pedestrian.yml'
        self.FLAGS.draw_threshold = 0.5
        self.FLAGS.opt = {'use_gpu': True,
                          'weights': '/D/MLProject/PoseTrack/LightTrackV2/weights/pp-predestrain/picodet_s_320_pedestrian.pdparams'}
        self.FLAGS.output_dir = 'output'
        cfg = load_config(self.FLAGS.config)
        cfg['use_vdl'] = self.FLAGS.use_vdl
        cfg['vdl_log_dir'] = self.FLAGS.vdl_log_dir
        merge_config(self.FLAGS.opt)

        # disable npu in config by default
        if 'use_npu' not in cfg:
            cfg.use_npu = False

        if cfg.use_gpu:
            place = paddle.set_device('gpu')
        elif cfg.use_npu:
            place = paddle.set_device('npu')
        else:
            place = paddle.set_device('cpu')

        if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
            cfg['norm_type'] = 'bn'

        if self.FLAGS.slim_config:
            cfg = build_slim_model(cfg, self.FLAGS.slim_config, mode='test')

        check_config(cfg)
        check_gpu(cfg.use_gpu)
        check_npu(cfg.use_npu)
        check_version()

        # build trainer
        self.trainer = Trainer(cfg, mode='test')

        # load weights
        self.trainer.load_weights(cfg.weights)

    # --model_dir=/D/MLProject/PoseTrack/LightTrackV2/weights/pp-predestrain/picodet_s_320_pedestrian --model_dir_keypoint=/D/MLProject/PoseTrack/LightTrackV2/weights/pp-tinypose/tinypose_128x96  --image_file=/D/tools/PaddleDetection/demo/000000014439.jpg --device=GPU

# -c /D/tools/PaddleDetection/configs/picodet/application/pedestrian_detection/picodet_s_320_pedestrian.yml -o use_gpu=true weights=https://bj.bcebos.com/v1/paddledet/models/keypoint/picodet_s_192_pedestrian.pdparams --infer_img=/D/tools/PaddleDetection/demo/000000014439.jpg
if __name__ == "__main__":
    phd = PaddleHumanDetector()
    phd.infer("/D/tools/PaddleDetection/demo/000000014439.jpg")
