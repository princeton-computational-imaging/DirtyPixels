"""Generic evaluation script that evaluates a model using the Dirty-Pixels captured dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import skimage.measure
import scipy.ndimage.filters
import tensorflow as tf
import scipy.io
import os
import cv2
from datasets import dataset_factory, build_imagenet_data
import numpy as np
from nets import nets_factory
from preprocessing import preprocessing_factory, sensor_model
import matplotlib.pyplot as plt
from pprint import pprint
from nets.isp import anscombe
import rawpy
import pyexifinfo

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'device', '0', 'GPU device to use.')

tf.app.flags.DEFINE_string(
    'sensor', 'Nexus_6P_rear', 'The sensor.')

tf.app.flags.DEFINE_string(
    'isp_model_name', None, 'The name of the ISP architecture to train.')

tf.app.flags.DEFINE_boolean('use_anscombe', True,
                            'Use Anscombe transform.')

tf.app.flags.DEFINE_boolean('noise_channel', True,
                            'Use noise channel.')

tf.app.flags.DEFINE_integer(
    'num_iters', 1,
    'Number of iterations for the unrolled Proximal Gradient Network.')

tf.app.flags.DEFINE_integer(
    'num_layers', 17, 'Number of layers to be used in the HQS ISP prior -- DEPRECATED')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')


FLAGS = tf.app.flags.FLAGS


def crop_and_subsample(img, target_size, average=None):
    factor = int(np.floor(min(img.shape) / target_size))
    ch = (img.shape[0] - factor * target_size) / 2
    cw = (img.shape[1] - factor * target_size) / 2
    cropped = img[int(np.floor(ch)):-int(np.ceil(ch)),
              int(np.floor(cw)):-int(np.ceil(cw))]
    if average is not None:
        cropped = scipy.ndimage.filters.convolve(cropped, np.ones((average, average)))
    return cropped[::factor, ::factor]


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        ####################
        # Select the model #
        ####################
        num_classes = 1001
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes,
            weight_decay=0.0,
            batch_norm_decay=0.95,
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        orig_image = tf.placeholder(tf.float32, shape=(eval_image_size, eval_image_size, 3))
        alpha = tf.placeholder(tf.float32, shape=[1, 3])
        sigma = tf.placeholder(tf.float32, shape=[1, 3])
        bayer_mask = sensor_model.get_bayer_mask(eval_image_size, eval_image_size)
        # image = image_preprocessing_fn(orig_image, orig_image, eval_image_size, eval_image_size, sensor=FLAGS.sensor)
        image = orig_image * bayer_mask
        # alpha, sigma = sensor_model.get_coeffs(light_level[None], sensor=FLAGS.sensor)
        # Scale to [-1, 1]
        if FLAGS.isp_model_name is None:
            image = 2 * (image - 0.5)

        images = tf.expand_dims(image, 0)

        ####################
        # Define the model #
        ####################
        inputs =  images

        network_ops = network_fn(images=inputs, alpha=alpha, sigma=sigma,
                                        bayer_mask=bayer_mask, use_anscombe=FLAGS.use_anscombe,
                                        noise_channel=FLAGS.noise_channel,
                                        num_classes=num_classes,
                                        num_iters=FLAGS.num_iters, num_layers=FLAGS.num_layers,
                                        isp_model_name=FLAGS.isp_model_name, is_real_data=True)
        logits, end_points = network_ops[:2]

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver()

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        synset2label = {}
        with open("datasets/synset_labels.txt", "r") as f:
            for line in f:
                synset, label = line.split(':')
                synset2label[synset] = int(label)

        if not os.path.isdir(FLAGS.eval_dir):
            os.mkdir(FLAGS.eval_dir)

        with tf.Session() as sess:
            saver.restore(sess, FLAGS.checkpoint_path)
            synsets = os.listdir(FLAGS.dataset_dir)
            number_to_human = {int(i[0]):i[1] for i in np.genfromtxt('datasets/imagenet_labels.txt', delimiter=':', dtype=np.string_)}

            # estimated alpha and gama 
            alpha_val = 0.0153 
            sigma_val = 0.0328 
            count = 0
            top1 = 0
            top5 = 0
            correct_paths = []
            wrong_paths = []
            for synset in synsets:
                if synset == 'labels.txt':
                    continue
                synset_top5 = 0
                path = os.path.join(FLAGS.dataset_dir, synset)
                image_names = [name for name in sorted(os.listdir(path)) if '.dng' in name]
                for imagename in image_names:
                    try:
                        loaded_image = rawpy.imread(os.path.join(path, imagename))
                        info = pyexifinfo.get_json(os.path.join(path, imagename))[0]
                        black_level = float(info['EXIF:BlackLevel'].split(' ')[0])
                        awb = [float(x) for x in info['EXIF:AsShotNeutral'].split(' ')]
                        raw_img = (loaded_image.raw_image_visible - black_level) / 1023.
                    except Exception as e:
                        print(synset, imagename, e)
                        continue

                    B = raw_img[::2, ::2] / awb[2]
                    R = raw_img[1::2, 1::2] / awb[0]
                    G1 = raw_img[1::2, ::2] / awb[1]
                    G2 = raw_img[::2, 1::2] / awb[1]
                    B, R, G1, G2 = (crop_and_subsample(img, eval_image_size // 2)
                                    for img in [B, R, G1, G2])
                    scale_factor = 1.0 / np.percentile(np.stack([B, R, G1, G2], axis=2), 98)

                    mosaiced = np.zeros((224, 224, 3))
                    mosaiced[::2, ::2, 2] = B
                    mosaiced[1::2, 1::2, 0] = R
                    mosaiced[1::2, ::2, 1] = G1
                    mosaiced[::2, 1::2, 1] = G2

                    img_scaled = mosaiced * scale_factor
                    input_img = np.clip(img_scaled, 0, 1)
                    scaling = (scale_factor / np.array(awb))[None, :]
                    logits_vals, clean_image = sess.run(
                        [logits[0, :], end_points.get('mobilenet_input', alpha)],
                        feed_dict={orig_image: input_img,
                                    alpha: alpha_val * scaling,
                                    sigma: sigma_val * scaling})
                    correct = synset2label[synset]
                    predictions = np.argsort(-logits_vals)
                    rank = np.nonzero(predictions == correct)[0]
                    clean_image = clean_image.squeeze()

                    if count % 100 == 0:
                        print("%d images out of 1000" % (count))

                    trgt_path = os.path.join(FLAGS.eval_dir, 'clean', synset)
                    raw_path = os.path.join(FLAGS.eval_dir, 'raw', synset)

                    if not os.path.exists(raw_path):
                        os.makedirs(raw_path)

                    if not os.path.exists(trgt_path):
                        os.makedirs(trgt_path)
                    cv2.imwrite(os.path.join(raw_path,  imagename[:-4]+'.png'), (input_img*255).astype(np.uint8))
                    if FLAGS.isp_model_name == 'isp':
                        trgt_path = os.path.join(trgt_path, imagename[:-4]+'.png')
                        plt.imsave(trgt_path, clean_image)

                    if rank == 0:
                        correct_paths.append("%s \"%s\" \"%s\""%(os.path.join(trgt_path, imagename[:-4]+'.png'), number_to_human[correct], number_to_human[predictions[0]]))
                        top1 += 1.0
                    else:
                        wrong_paths.append("%s \"%s\" \"%s\""%(os.path.join(trgt_path, imagename[:-4]+'.png'), number_to_human[correct], number_to_human[predictions[0]]))

                    if rank <= 5:
                        top5 += 1.0
                        synset_top5 += 1.0
                    count += 1

                print("Synset %s, Top 5 %f" % (synset, synset_top5 / len(image_names)))

            print("Top-1 %f, Top-5 %f" % (top1 / count, top5 / count))

            with open(os.path.join(FLAGS.eval_dir, 'correct.txt'), 'w') as f:
                for item in correct_paths:
                    f.write("%s\n" % item)
            with open(os.path.join(FLAGS.eval_dir, 'wrong.txt'), 'w') as f:
                for item in wrong_paths:
                    f.write("%s\n" % item)


if __name__ == '__main__':
    tf.app.run()



