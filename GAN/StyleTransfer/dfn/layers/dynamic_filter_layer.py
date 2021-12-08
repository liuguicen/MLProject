# -*- coding: utf-8 -*-


import numpy as np
from collections import OrderedDict
import sys
import os

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.base import Layer, MergeLayer

from lasagne.layers.conv import conv_output_length
from lasagne.layers.pool import pool_output_length
from lasagne.utils import as_tuple

from theano.sandbox.cuda import dnn # xu

__all__ = [
    "DynamicFilterLayer"
]

class DynamicFilterLayer(MergeLayer):
    def __init__(self, incomings, filter_size, stride=1, pad=0, flip_filters=False, grouping=False, **kwargs):
        super(DynamicFilterLayer, self).__init__(incomings, **kwargs)

        self.filter_size = lasagne.utils.as_tuple(filter_size, 3, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.flip_filters = flip_filters
        self.grouping = grouping

        if self.grouping:
            assert(filter_size[2] == 1)

    def get_output_shape_for(self, input_shapes):
        if self.grouping:
            shape = (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2], input_shapes[0][3])
        else:
            shape = (input_shapes[0][0], 1, input_shapes[0][2], input_shapes[0][3])
        return shape

    def get_output_for(self, input, **kwargs):
        image = input[0]
        filters = input[1] # 这个就是得出来的过滤器参数

        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        filter_size = self.filter_size

        if self.grouping:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (np.prod(filter_size), 1, filter_size[0], filter_size[1]))
            filter_localexpand = T.cast(theano.shared(filter_localexpand_np), 'floatX')

            outputs = []
            for i in range(3):
                input_localexpanded = dnn.dnn_conv(img=image[:,[i],:,:], kerns=filter_localexpand, subsample=self.stride, border_mode=border_mode, conv_mode=conv_mode)
                output = T.sum(input_localexpanded * filters, axis=1, keepdims=True)
                outputs.append(output)

            output = T.concatenate(outputs, axis=1)
        else:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (np.prod(filter_size), filter_size[2], filter_size[0], filter_size[1]))
            filter_localexpand = T.cast(theano.shared(filter_localexpand_np), 'floatX')
            input_localexpanded = dnn.dnn_conv(img=image, kerns=filter_localexpand, subsample=self.stride, border_mode=border_mode, conv_mode=conv_mode)
            output = input_localexpanded * filters
            output = T.sum(output, axis=1, keepdims=True)

        return output