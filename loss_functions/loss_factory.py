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

import tensorflow as tf

from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import lenet_preprocessing
from preprocessing import vgg_preprocessing

slim = tf.contrib.slim



def get_loss(name):
  """Returns loss_fn(outputs, ground_truths, **kwargs), where "outputs" are the model outputs.

  Args:
    name: The name of the loss function.

  Returns:
    loss_fn: A function that computes the loss between the inputs and the ground_truths

  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  loss_fn_map = {
    'mean_squared_error':slim.losses.mean_squared_error,
    'absolute_difference':slim.losses.absolute_difference
  }

  if name not in loss_fn_map:
    raise ValueError('Loss function name [%s] was not recognized' % name)

  def loss_fn(outputs, ground_truths, **kwargs):
    return loss_fn_map[name](
        outputs, ground_truths, **kwargs)

  return loss_fn
