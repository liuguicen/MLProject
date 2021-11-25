from torch import nn
from torch.nn import functional as F

import MRCnnConfig as config


class RPNHead(nn.Module):

    def __init__(self, inchannel):
        nn.Module.__init__(self)
        # self.conv_anchor_view_expand = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1,
        #                                          padding=1)
        # self.conv_box_score = nn.Conv2d(in_channels=inchannel, out_channels=config.ANCHORS_PER_LOCATION, kernel_size=1,
        #                                 stride=1)
        #
        # self.conv_box_delta = nn.Conv2d(in_channels=inchannel, out_channels=4 * config.ANCHORS_PER_LOCATION,
        #                                 kernel_size=1,
        #                                 stride=1)
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=inchannel, kernel_size=3, stride=1,
                              padding=1)
        self.cls_logits = nn.Conv2d(in_channels=inchannel, out_channels=config.ANCHORS_PER_LOCATION, kernel_size=1,
                                    stride=1)

        self.bbox_pred = nn.Conv2d(in_channels=inchannel, out_channels=4 * config.ANCHORS_PER_LOCATION,
                                   kernel_size=1,
                                   stride=1)

    def forward(self, x):
        box_score_list = []
        box_delta_list = []
        for feature_layer in x:
            # 卷积+relu, 扩大anchor点的视野，综合预测锚框和其分数
            feature_layer = self.conv(feature_layer)
            feature_layer = F.relu(feature_layer)
            # 第一条路，获取正负框得分
            box_score_list.append(self.cls_logits(feature_layer))
            # 第二条路，获取框的调整参数
            box_delta_list.append(self.bbox_pred(feature_layer))
        return box_score_list, box_delta_list
