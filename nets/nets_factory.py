# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import isp
from nets import mobilenet_v1
from nets import mobilenet_isp

slim = tf.contrib.slim

networks_map = {'isp': isp.prox_grad_isp,
                'mobilenet_isp': mobilenet_isp.mobilenet_v1,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'deeper_mobilenet_v1': mobilenet_v1.deeper_mobile_net_v1,
               }

arg_scopes_map = {'isp': isp.isp_arg_scope,
                  'mobilenet_isp': mobilenet_isp.mobilenet_v1_arg_scope,
                  'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'deeper_mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay, batch_norm_decay, is_training):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)

  func = networks_map[name]

  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)

    with slim.arg_scope(arg_scope):
      return func(images, is_training=is_training, **kwargs)

  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
