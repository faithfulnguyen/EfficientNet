from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import tensorflow as tf
import tensorflow.contrib.slim as slim


DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def se_block(input_feature, name, ratio=16):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')

        scale = input_feature * excitation

    return scale

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return slim.conv2d(
            inputs, num_outputs, kernel_size, stride=1, rate=rate,
            padding='SAME', scope=scope, activation_fn=tf.nn.swish
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(
            inputs, num_outputs, kernel_size, stride=stride,
            rate=rate, padding='VALID', scope=scope, activation_fn=None
        )


def conv2d_same_depthwise(inputs, kernel_size, stride, rate=1, scope=None):
    in_channels = inputs.get_shape()[-1]
    channel_multiplier = 1

    depthwise_filter = tf.get_variable(shape=(kernel_size, kernel_size, in_channels, channel_multiplier),
                                       name="deptwise_filter")

    if stride == 1:
        return tf.nn.depthwise_conv2d(
            inputs, depthwise_filter, strides=[1, 1, 1, 1], padding="SAME", name=scope + "depth_wise"
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        return tf.nn.depthwise_conv2d(
            inputs, depthwise_filter, strides= [1, stride, stride, 1] , padding="VALID", name= scope + "depth_wise"

        )

def block(inputs,
          phase_train=True,
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):

    # Expansion phase
    with tf.variable_scope(name, 'bottleneck_v2', [inputs]) as sc:
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            net = slim.conv2d(
                inputs, filters, 1, stride=1,
                padding='SAME', scope=name + 'expand_conv', activation_fn=tf.nn.swish
            )
        else:
            net = inputs

        net = conv2d_same_depthwise(net, kernel_size, stride=strides, rate=1, scope=name)
        net = slim.batch_norm(net, scope='bn_' + name)
        net = tf.nn.swish(net)

    if (0.0 < se_ratio and se_ratio <= 1.0):
        net = se_block(net, name + '_se_block')

    net = slim.conv2d(
        net, filters_out, 1, stride=1,
        padding='SAME', scope=name + 'project_conv', activation_fn=None
    )
    net = tf.nn.swish(net)

    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            slim.dropout(net, keep_prob=1 - drop_rate, is_training=phase_train, scope=name + 'drop')
        net += inputs
    return net


def EfficientNet(
        img_input,
        phase_train,
        width_coefficient,
        depth_coefficient,
        drop_connect_rate=0.2,
        depth_divisor=8,
        blocks_args='default'):
    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    net = conv2d_same(img_input, round_filters(32), 3, stride=2, rate=1, scope=None)
    net = slim.batch_norm(net, scope="bn_1")
    net = tf.nn.swish(net)

    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']

            net = block(
                net, phase_train,
                drop_connect_rate * b / blocks,
                name='block{}{}_'.format(i + 1, chr(j + 97)),
                **args)
            b += 1
            print(net)

    # Build top
    net = slim.conv2d(
        net, round_filters(1280), 1, stride=1,
        padding='SAME', scope="top_convd", activation_fn=None
    )
    net = slim.batch_norm(net, scope="bn_top")
    net = tf.nn.swish(net)
    return net


def EfficientNetB4(images, bottleneck_layer_size, dropout_rate, phase_train):
    net = EfficientNet(
        images, phase_train, width_coefficient=1.4, depth_coefficient=1.8
    )
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=False)
    net = slim.dropout(net, keep_prob=1.0 - dropout_rate, is_training=phase_train, scope='dropout_fn')
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck',
                               reuse=False)
    return net

def EfficientNetB7(images, bottleneck_layer_size, dropout_rate, phase_train):
    net = EfficientNet(
        images, phase_train, width_coefficient=2, depth_coefficient=3.1
    )
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=False)
    net = slim.dropout(net, keep_prob=1.0 - dropout_rate, is_training=phase_train, scope='dropout_fn')
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck',
                               reuse=False)
    return net


def EfficientNetBXXX_TestGit(images, bottleneck_layer_size, dropout_rate, phase_train):
    net = EfficientNet(
        images, phase_train, width_coefficient=4, depth_coefficient=4.4
    )
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=False)
    net = slim.dropout(net, keep_prob=1.0 - dropout_rate, is_training=phase_train, scope='dropout_fn')
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck',
                               reuse=False)
    return net


def EfficientNetBXXX_TestGit_Branch(images, bottleneck_layer_size, dropout_rate, phase_train):
    net = EfficientNet(
        images, phase_train, width_coefficient=4, depth_coefficient=4.4
    )
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=False)
    net = slim.dropout(net, keep_prob=1.0 - dropout_rate, is_training=phase_train, scope='dropout_fn')
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck',
                               reuse=False)
    return net


import numpy as np
def test():
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    image_plh = tf.placeholder(tf.float32, shape=(None, 64, 256, 3), name='img_data_1')
    net = EfficientNetB4(image_plh, bottleneck_layer_size=128, dropout_rate=0.4, phase_train=True)
    total_number_parameter = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("total_number_parameter ", total_number_parameter)

if __name__ == "__main__":
    test()
