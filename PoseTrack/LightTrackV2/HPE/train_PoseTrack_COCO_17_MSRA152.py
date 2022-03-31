'''
    Modified by: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    March, 2019
'''
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim
import sys, os
import argparse
import numpy as np
from functools import partial

from config import cfg
from tfflat.base import ModelDesc, Trainer
from tfflat.utils import mem_info

from nets.basemodel import resnet152, resnet_arg_scope, resnet_v1
resnet_arg_scope = partial(resnet_arg_scope, bn_trainable=cfg.bn_train)

def create_global_net(blocks, is_training, trainable=True):
    global_fms = []
    global_outs = []
    last_fm = None
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    for i, block in enumerate(reversed(blocks)):
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            lateral = slim.conv2d(block, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='lateral/res{}'.format(5-i))

        if last_fm is not None:
            sz = tf.shape(input=lateral)
            upsample = tf.image.resize(last_fm, (sz[1], sz[2]),
                method=tf.image.ResizeMethod.BILINEAR, name='upsample/res{}'.format(5-i))
            upsample = slim.conv2d(upsample, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=None,
                scope='merge/res{}'.format(5-i))
            last_fm = upsample + lateral
        else:
            last_fm = lateral

        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            tmp = slim.conv2d(last_fm, 256, [1, 1],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='tmp/res{}'.format(5-i))
            out = slim.conv2d(tmp, cfg.nr_skeleton, [3, 3],
                trainable=trainable, weights_initializer=initializer,
                padding='SAME', activation_fn=None,
                scope='pyramid/res{}'.format(5-i))
        global_fms.append(last_fm)
        global_outs.append(tf.image.resize(out, (cfg.output_shape[0], cfg.output_shape[1]), method=tf.image.ResizeMethod.BILINEAR))
    global_fms.reverse()
    global_outs.reverse()
    return global_fms, global_outs

def create_refine_net(blocks, is_training, trainable=True):
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    bottleneck = resnet_v1.bottleneck
    refine_fms = []
    for i, block in enumerate(blocks):
        mid_fm = block
        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
            for j in range(i):
                mid_fm = bottleneck(mid_fm, 256, 128, stride=1, scope='res{}/refine_conv{}'.format(2+i, j)) # no projection
        mid_fm = tf.image.resize(mid_fm, (cfg.output_shape[0], cfg.output_shape[1]),
            method=tf.image.ResizeMethod.BILINEAR, name='upsample_conv/res{}'.format(2+i))
        refine_fms.append(mid_fm)
    refine_fm = tf.concat(refine_fms, axis=3)
    with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):
        refine_fm = bottleneck(refine_fm, 256, 128, stride=1, scope='final_bottleneck')
        res = slim.conv2d(refine_fm, cfg.nr_skeleton, [3, 3],
            trainable=trainable, weights_initializer=initializer,
            padding='SAME', activation_fn=None,
            scope='refine_out')
    return res


class Network(ModelDesc):

    def render_gaussian_heatmap(self, coord, output_shape, sigma):

        x = [i for i in range(output_shape[1])]
        y = [i for i in range(output_shape[0])]
        xx,yy = tf.meshgrid(x,y)
        xx = tf.reshape(tf.cast(xx, dtype=tf.float32), (1,*output_shape,1))
        yy = tf.reshape(tf.cast(yy, dtype=tf.float32), (1,*output_shape,1))

        x = tf.floor(tf.reshape(coord[:,:,0],[-1,1,1,cfg.nr_skeleton]) / cfg.data_shape[1] * output_shape[1] + 0.5)
        y = tf.floor(tf.reshape(coord[:,:,1],[-1,1,1,cfg.nr_skeleton]) / cfg.data_shape[0] * output_shape[0] + 0.5)

        heatmap = tf.exp(-(((xx-x)/tf.cast(sigma, dtype=tf.float32))**2)/tf.cast(2, dtype=tf.float32) -(((yy-y)/tf.cast(sigma, dtype=tf.float32))**2)/tf.cast(2, dtype=tf.float32))
        return heatmap * 255.

    def make_data(self):
        ''' Load PoseTrack data '''
        from AllJoints_PoseTrack import PoseTrackJoints
        from AllJoints_COCO import PoseTrackJoints_COCO
        from dataset import Preprocessing

        d = PoseTrackJoints()
        train_data, _ = d.load_data(cfg.min_kps)
        print(len(train_data))

        #'''
        d2 = PoseTrackJoints_COCO()
        train_data_coco, _ = d2.load_data(cfg.min_kps)
        print(len(train_data_coco))

        train_data.extend(train_data_coco)
        print(len(train_data))
        #'''

        from random import shuffle
        shuffle(train_data)

        from tfflat.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        dp = DataFromList(train_data)
        if cfg.dpflow_enable:
            dp = MultiProcessMapDataZMQ(dp, cfg.nr_dpflows, Preprocessing)
        else:
            dp = MapData(dp, Preprocessing)
        dp = BatchData(dp, cfg.batch_size // cfg.nr_aug)
        dp.reset_state()
        dataiter = dp.get_data()

        return dataiter

    def head_net(self, blocks, is_training, trainable=True):

        normal_initializer = tf.compat.v1.truncated_normal_initializer(0, 0.01)
        msra_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
        xavier_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

        with slim.arg_scope(resnet_arg_scope(bn_is_training=is_training)):

            out = slim.conv2d_transpose(blocks[-1], 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up1')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up2')
            out = slim.conv2d_transpose(out, 256, [4, 4], stride=2,
                trainable=trainable, weights_initializer=normal_initializer,
                padding='SAME', activation_fn=tf.nn.relu,
                scope='up3')

            out = slim.conv2d(out, cfg.nr_skeleton, [1, 1],
                    trainable=trainable, weights_initializer=msra_initializer,
                    padding='SAME', normalizer_fn=None, activation_fn=None,
                    scope='out')

        return out


    def make_network(self, is_train):
        if is_train:
            image = tf.compat.v1.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.data_shape, 3])
            #target_coord = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.nr_skeleton, 2])
            #valid = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.nr_skeleton])
            #self.set_inputs(image, target_coord, valid)

            label15 = tf.compat.v1.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label11 = tf.compat.v1.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label9 = tf.compat.v1.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            label7 = tf.compat.v1.placeholder(tf.float32, shape=[cfg.batch_size, *cfg.output_shape, cfg.nr_skeleton])
            valids = tf.compat.v1.placeholder(tf.float32, shape=[cfg.batch_size, cfg.nr_skeleton])
            labels = [label15, label11, label9, label7]
            # labels.reverse() # The original labels are reversed. For reproduction of our pre-trained model, I'll keep it same.
            self.set_inputs(image, label15, label11, label9, label7, valids)
        else:
            image = tf.compat.v1.placeholder(tf.float32, shape=[None, *cfg.data_shape, 3])
            self.set_inputs(image)

        resnet_fms = resnet152(image, is_train, bn_trainable=True)

        heatmap_outs = self.head_net(resnet_fms, is_train)

        # make loss
        if is_train:
            def ohkm(loss, top_k):
                ohkm_loss = 0.
                for i in range(cfg.batch_size):
                    sub_loss = loss[i]
                    topk_val, topk_idx = tf.nn.top_k(sub_loss, k=top_k, sorted=False, name='ohkm{}'.format(i))
                    tmp_loss = tf.gather(sub_loss, topk_idx, name='ohkm_loss{}'.format(i)) # can be ignore ???
                    ohkm_loss += tf.reduce_sum(input_tensor=tmp_loss) / top_k
                ohkm_loss /= cfg.batch_size
                return ohkm_loss

            label = label7 * tf.cast(tf.greater(tf.reshape(valids, (-1, 1, 1, cfg.nr_skeleton)), 0.1), dtype=tf.float32)
            loss = tf.reduce_mean(input_tensor=tf.square(heatmap_outs - label))

            self.add_tower_summary('loss', loss)
            self.set_loss(loss)
        else:
            self.set_outputs(heatmap_outs)


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', '-d', type=str, dest='gpu_ids')
        parser.add_argument('--continue', '-c', dest='continue_train', action='store_true')
        parser.add_argument('--debug', dest='debug', action='store_true')
        args = parser.parse_args()

        if not args.gpu_ids:
            args.gpu_ids = str(np.argmin(mem_info()))

        if '-' in args.gpu_ids:
            gpus = args.gpu_ids.split('-')
            gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
            gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
            args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

        return args
    args = parse_args()

    cfg.set_args(args.gpu_ids, args.continue_train)
    trainer = Trainer(Network(), cfg)
    trainer.train()