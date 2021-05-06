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


# This file evaluates a trained network on a test dataset and saves the filenames of images
# that were correctly / falsely classified into a text file, so that the images that different
# classifiers got right / wrong can be compared.

"""Generic evaluation script that evaluates a model using a given dataset.
  Noise is introduced before images are input to the classifier, and it is defined by the 
  mode parameter, and the Camera Image Formation
  model defined in the Dirty-Pixels manuscript.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
from glob import glob

import cv2

from preprocessing import preprocessing_factory, sensor_model
from datasets import dataset_factory
from nets import nets_factory
import numpy as np

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
            'device', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
            'mode', '3lux', 'Noise profile: 3lux, 6lux, 2to20lux, or 2to200lux.')


tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', None, 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string('eval_dir', 'output_synthetic_images', 'Output directory')

FLAGS = tf.app.flags.FLAGS


def imnet_generator(root_directory):
    # list all directories
    dirs = sorted(glob(os.path.join(root_directory, "*/")))
    print("#### num dirs", len(dirs))

    # Build the label lookup table
    synset_to_label = {synset.decode('utf-8'):i+1 for i, synset in enumerate(np.genfromtxt('datasets/imagenet_lsvrc_2015_synsets.txt', dtype=np.string_))}
    # print(synset_to_label.items())

    # loop through directories and glob all images
    for idx, dir in enumerate(dirs):
        # Glob all image files in this directory
        img_files = glob(os.path.join(dir, '*.png'))
        img_files += glob(os.path.join(dir, '*.jpg'))
        img_files += glob(os.path.join(dir, '*.jpeg'))
        img_files += glob(os.path.join(dir, '*.JPEG'))

        for img_file in img_files:
            yield img_file, synset_to_label[os.path.basename(os.path.normpath(dir))], os.path.basename(os.path.normpath(dir))

def parse_img(img_path):
    rgb_string = tf.read_file(img_path)
    rgb_decoded = tf.image.decode_jpeg(rgb_string) # uint8
    rgb_decoded = tf.cast(rgb_decoded, tf.float32)
    rgb_decoded /= 255.
    return rgb_decoded

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device
  eval_dir = FLAGS.eval_dir

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    num_classes = 1001
    eval_image_size = 128

    image_path_graph = tf.placeholder(tf.string)
    label_graph = tf.placeholder(tf.int32)

    image = parse_img(image_path_graph)

    image = tf.image.central_crop(image, central_fraction=0.875)

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_images(image, [eval_image_size, eval_image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.squeeze(image, [0])

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=num_classes,
        batch_norm_decay=0.9,
        weight_decay=0.0,
        is_training=False)

    image.set_shape([128,128,3])
    image = tf.expand_dims(image, 0)

    if FLAGS.mode == '2to20lux':
        ll_low = 0.001
        ll_high = 0.01
    elif FLAGS.mode == '2to200lux':
        ll_low = 0.001
        ll_high = 0.1
    elif FLAGS.mode == '3lux':
        ll_low = 0.0015
        ll_high = 0.0015
    elif FLAGS.mode == '6lux':
        ll_low = 0.003
        ll_high = 0.003

    noisy_batch, alpha, sigma = \
                sensor_model.sensor_noise_rand_light_level(image, [ll_low, ll_high], scale=1.0, sensor='Nexus_6P_rear')
    bayer_mask = sensor_model.get_bayer_mask(128, 128)

    raw_image_graph = noisy_batch * bayer_mask

    ####################
    # Define the model #
    ####################
    logits, end_points, cleaned_image_graph = network_fn(images=raw_image_graph, alpha=alpha, sigma=sigma,
                bayer_mask=bayer_mask, use_anscombe=True,
                noise_channel=True,
                num_classes=num_classes,
                num_iters=1, num_layers=17,
                isp_model_name='isp')

    predictions = tf.argmax(logits, 1)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        print('###### Loading last checkpoint of directory', FLAGS.checkpoint_path)
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        print('###### Loading checkpoint', FLAGS.checkpoint_path)
        checkpoint_path = FLAGS.checkpoint_path


    tf.logging.info('Evaluating %s' % FLAGS.checkpoint_path)

    correct_paths = []
    wrong_paths = []

    # Restore variables from checkpoint
    variables_to_restore = slim.get_variables_to_restore() # slim.get_model_variables() 
    saver = tf.train.Saver(variables_to_restore)

    number_to_human = {int(i[0]):i[1] for i in np.genfromtxt('datasets/imagenet_labels.txt', delimiter=':', dtype=np.string_)}

    eval_dir= FLAGS.eval_dir
    os.makedirs(eval_dir, exist_ok=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_path)

        count = 0
        for img_file, label, synset in imnet_generator(FLAGS.dataset_dir):
            preds_value, cleaned_image, raw_image = sess.run([predictions, cleaned_image_graph, raw_image_graph],
                    feed_dict={image_path_graph:img_file, label_graph:label})

            cleaned_image = np.clip(cleaned_image, 0.0, 1.0).squeeze()[:,:,::-1]
            raw_image = raw_image.squeeze()[:,:,::-1]
            img_filename = os.path.basename(os.path.normpath(img_file))

            our_path = os.path.join(eval_dir, 'anscombe_output', FLAGS.mode, synset)
            raw_path = os.path.join(eval_dir, 'raw', FLAGS.mode, synset)

            if not os.path.exists(our_path):
                os.makedirs(our_path)
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)

            if count % 10000 == 0:
                print('num. processed ', count)
                print('num. correct paths', len(correct_paths))
            count += 1
            img_filename = os.path.splitext(img_filename)[0] + '.png'

            cv2.imwrite(os.path.join(our_path, img_filename), (cleaned_image*255).astype(np.uint8))
            cv2.imwrite(os.path.join(raw_path, img_filename), (raw_image*255).astype(np.uint8))

            if preds_value.squeeze() == label:
                correct_paths.append("%s \"%s\" \"%s\""%(os.path.join(our_path, img_filename), number_to_human[label], number_to_human[preds_value[0]]))
            else:
                wrong_paths.append("%s \"%s\" \"%s\""%(os.path.join(our_path, img_filename), number_to_human[label], number_to_human[preds_value[0]]))

        print('Top-1 accuracy', float(len(correct_paths))/float(len(wrong_paths)+len(correct_paths)))
    correct_paths_fn = os.path.join(eval_dir, FLAGS.mode + '_correct.txt')
    with open(correct_paths_fn, 'w') as f:
        for item in correct_paths:
            f.write("%s\n" % item)
    wrong_paths_fn = os.path.join(eval_dir, FLAGS.mode + '_wrong.txt')
    with open(wrong_paths_fn, 'w') as f:
        for item in wrong_paths:
            f.write("%s\n" % item)

if __name__ == '__main__':
  tf.app.run()
