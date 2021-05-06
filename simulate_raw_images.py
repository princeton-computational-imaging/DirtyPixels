"""Script for adding noise to ImageNet-like dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
import cv2
from datasets import dataset_factory, build_imagenet_data
import numpy as np
from preprocessing import preprocessing_factory, sensor_model
from pprint import pprint
from glob import glob


tf.app.flags.DEFINE_float(
     'll_low', None,
     'Lowest light level.')

tf.app.flags.DEFINE_float(
     'll_high', None,
     'Highest light level.')

tf.app.flags.DEFINE_string(
     'sensor', 'Nexus_6P_rear', 'The sensor.')

tf.app.flags.DEFINE_string(
    'output_dir', None, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'input_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'mobilenet_v1', 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 128, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def main(_):
  if not FLAGS.input_dir:
    raise ValueError('You must supply the input directory with --input_dir')
  if not FLAGS.output_dir:
    raise ValueError('You must supply the dataset directory with --output_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():    

    # Preprocess the images so that they all have the same size
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size
    orig_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
    image = image_preprocessing_fn(orig_image, orig_image, eval_image_size, eval_image_size)
    images = tf.expand_dims(image, 0)

    # Add noise.
    noisy_batch, alpha, sigma = sensor_model.sensor_noise_rand_light_level(images,
                                [FLAGS.ll_low, FLAGS.ll_high],
                                scale=1.0, sensor=FLAGS.sensor)

    bayer_mask = sensor_model.get_bayer_mask(eval_image_size, eval_image_size)
    inputs = noisy_batch*bayer_mask

    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    with tf.Session() as sess:
        count = 0
        synsets = [path for path in os.listdir(FLAGS.input_dir) if not '.' in path]

        for synset in synsets:
            path = os.path.join(FLAGS.input_dir, synset)
            image_names = os.listdir(path)
            print("Found %d images in %s"%(len(image_names), synset))

            synset_path = os.path.join(FLAGS.output_dir, synset)
            if not os.path.isdir(synset_path):
                os.mkdir(synset_path)

            for imagename in image_names:
                output_imgfn = os.path.join(FLAGS.output_dir, synset, imagename.split('.')[0]+'.png')
                if os.path.isfile(output_imgfn):
                    continue
                loaded_image = cv2.imread(os.path.join(path, imagename))
                
                # BGR to RGB
                loaded_image = loaded_image[..., ::-1]
                images, alpha_val, sigma_val = sess.run(
                        [inputs, alpha, sigma],
                        feed_dict={orig_image:loaded_image})
                img = (255.0*images[0,:,:,:]).astype(np.uint8)

                # RGB to BGR
                img = img[..., ::-1]

                if count % 1000 == 0:
                    print("%d processed images." % (count))
                cv2.imwrite(output_imgfn, img)
                count += 1

    print('Total images processed:', count)


if __name__ == '__main__':
  tf.app.run()
