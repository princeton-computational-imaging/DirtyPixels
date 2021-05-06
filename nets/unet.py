# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf

slim = tf.contrib.slim

def lrelu(x):
  return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
  pool_size = 2
  deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
  deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

  deconv_output = tf.concat([deconv, x2], 3)
  deconv_output.set_shape([None, None, None, output_channels * 2])

  return deconv_output


def unet(input, scope=None):
  with tf.variable_scope(scope, 'gauss_den_chen_unet', [input]) as sc:
    conv1 = slim.conv2d(input, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 64, 128)
    conv6 = slim.conv2d(up6, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 32, 64)
    conv7 = slim.conv2d(up7, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 16, 32)
    conv8 = slim.conv2d(up8, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 8, 16)
    conv9 = slim.conv2d(up9, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    # conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    # out = tf.depth_to_space(conv10, 2)
    out = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
  return out

# def unet(input, scope=None):
#   with tf.variable_scope(scope, 'gauss_den_chen_unet', [input]) as sc:
#     conv1 = slim.conv2d(input, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
#     conv1 = slim.conv2d(conv1, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
#     pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

#     conv2 = slim.conv2d(pool1, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
#     conv2 = slim.conv2d(conv2, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
#     pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

#     conv3 = slim.conv2d(pool2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
#     conv3 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
#     pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

#     conv4 = slim.conv2d(pool3, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
#     conv4 = slim.conv2d(conv4, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
#     pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

#     conv5 = slim.conv2d(pool4, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
#     conv5 = slim.conv2d(conv5, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

#     up6 = upsample_and_concat(conv5, conv4, 32, 64)
#     conv6 = slim.conv2d(up6, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
#     conv6 = slim.conv2d(conv6, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

#     up7 = upsample_and_concat(conv6, conv3, 32, 64)
#     conv7 = slim.conv2d(up7, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
#     conv7 = slim.conv2d(conv7, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

#     up8 = upsample_and_concat(conv7, conv2, 16, 32)
#     conv8 = slim.conv2d(up8, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
#     conv8 = slim.conv2d(conv8, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

#     up9 = upsample_and_concat(conv8, conv1, 8, 16)
#     conv9 = slim.conv2d(up9, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
#     conv9 = slim.conv2d(conv9, 8, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

#     # conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
#     # out = tf.depth_to_space(conv10, 2)
#     out = slim.conv2d(conv9, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
#   return out