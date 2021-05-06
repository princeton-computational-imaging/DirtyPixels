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

from preprocessing import inception_preprocessing
from preprocessing import isp_pretrain_preprocessing
from preprocessing import joint_isp_preprocessing
from preprocessing import writeout_preprocessing
from preprocessing import no_preprocessing

slim = tf.contrib.slim


def get_preprocessing(name, is_training):
  """Returns preprocessing_fn(image, height, width, **kwargs).

  Args:
    name: The name of the preprocessing function.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    preprocessing_fn: A function that preprocessing a single image (pre-batch).
      It has the following signature:
        image = preprocessing_fn(image, output_height, output_width, ...).

  Raises:
    ValueError: If Preprocessing `name` is not recognized.
  """
  preprocessing_fn_map = {
      'isp': isp_pretrain_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'mobilenet_isp': joint_isp_preprocessing,
      'resnet_isp': isp_pretrain_preprocessing,
      'gharbi_isp': isp_pretrain_preprocessing,
      'writeout': writeout_preprocessing,
      'none': no_preprocessing,
      'deeper_mobilenet_v1': inception_preprocessing,
  }

  if name not in preprocessing_fn_map:
    raise ValueError('Preprocessing name [%s] was not recognized' % name)

  def preprocessing_fn(image, ground_truth, output_height, output_width, **kwargs):
    return preprocessing_fn_map[name].preprocess_image(
        image, ground_truth, output_height, output_width, is_training=is_training, **kwargs)

  return preprocessing_fn
