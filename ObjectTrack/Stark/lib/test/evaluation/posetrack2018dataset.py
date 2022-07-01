import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob
import six


class PoseTrack2018Dataset(BaseDataset):
    """
    TC-128 Dataset (78 newly added sequences)
    modified from the implementation in got10k-toolkit (https://github.com/got-10k/toolkit)
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.posetrack2018_path
        self.sequence_list = os.listdir(self.base_path)
        '''应该是每个视频的类别名字'''
        # self.clean_list = self.clean_seq_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        '''
        顾名思义，构造序列，就是构造用于模型处理的视频的序列数据对象，序列中的元素以视频帧为中心，包含相关的各种信息，如帧路径，对象框位置，对象是否消失等信息
        '''
        class_name = sequence_name.split('-')[0]
        anno_path = '/E/dataset/PoseTrack/PoseTrack2018/annotations/val/{}.json'.format(class_name)
        # 目标框以文本的形式，按行排列，直接加载进来
        ground_truth_rect = loadAnno(anno_path)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

import json
def loadAnno(path):
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
    load_dict['smallberg'] = [8200, {1: [['Python', 81], ['shirt', 300]]}]

def __len__(self):
        return len(self.seq_names)
