"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import my_util.util as util
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


# 通过内置的add_module方法添加多个判别器
class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)  # 进行一个下采样，然后再次送入判别器

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    '''
    单个判别器还是比较简单的，就是堆叠若干层 卷积 归一化 激活 就完了
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kernel_size = 4
        pad_w = int(np.ceil((kernel_size - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kernel_size, stride=2, padding=pad_w),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kernel_size,
                                               stride=stride, padding=pad_w)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=1, padding=pad_w)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))  # 保存每一层的函数对象

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
