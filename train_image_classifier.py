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
"""Generic training script that trains a model using a given dataset.
  Noise is introduced before images are input to the classifier, and it is defined by the ll_low, and ll_high
  parameters (lowest and highest noise levels), and the Camera Image Formation
  model defined in the Dirty-Pixels manuscript.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory, sensor_model
from pprint import pprint
import os

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None,
    'Size of training images')

tf.app.flags.DEFINE_string(
    'isp_checkpoint_path', None,
    'Path to the checkpoint of the pretrained ISP')

tf.app.flags.DEFINE_integer(
    'num_layers', 17,
    'Number of layers to be used in the HQS ISP prior -- DEPRECATED')

tf.app.flags.DEFINE_integer(
    'num_iters', 1,
    'Number of iterations for the unrolled Proximal Gradient Network')

tf.app.flags.DEFINE_float(
    'll_low', None,
    'Lowest light level.')

tf.app.flags.DEFINE_float(
    'll_high', None,
    'Highest light level.')

tf.app.flags.DEFINE_string('device', '0',
                'GPU device')

tf.app.flags.DEFINE_boolean('use_anscombe', True,
                            'Use Anscombe transform.')

tf.app.flags.DEFINE_boolean('noise_channel', True,
                            'Use noise channel.')

tf.app.flags.DEFINE_integer('num_clones', 4,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 7200,
    'The frequency with which the model is saved, in seconds.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.00045, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 1.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', 0.95,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'isp_model_name', 'gharbi_isp', 'The name of ISP architecture to train.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v4', 'The name of the Classification architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / (FLAGS.batch_size * FLAGS.num_clones) *
                    FLAGS.num_epochs_per_decay)

  return tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True,
                                    name='exponential_decay_learning_rate')


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  optimizer = tf.train.RMSPropOptimizer(
      learning_rate,
      decay=FLAGS.rmsprop_decay,
      momentum=FLAGS.momentum,
      epsilon=FLAGS.opt_epsilon)

  return optimizer


def _init_isp_func():
    all_vars = tf.all_variables()
    #all_vars = slim.get_variables_to_restore()

    isp_vars = [var for var in all_vars if 'MobilenetV1' not in var.name and 'ExponentialMovingAverage' not in var.name \
                and 'RMSProp' not in var.name and 'global_step' not in var.name]
    print("\n\nISP names:")
    pprint([var.name for var in isp_vars])

    saver = tf.train.Saver(isp_vars)

    def callback(session):
        saver.restore(session, FLAGS.isp_checkpoint_path)
    return callback


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  if FLAGS.model_name=='mobilenet_isp':
     if FLAGS.isp_checkpoint_path is not None:
         isp_init_func = _init_isp_func()
         variables_to_restore = [var for var in variables_to_restore if 'MobilenetV1' in var.name]
         print("Restoring isp variables and inception variables from seperate checkpoints.")

  #        print("\nInception vars:")
  #        pprint([var.name for var in variables_to_restore])
  #    else:
  #        print("Restoring all variables from a single checkpoint.")
  #        pprint([var.name for var in variables_to_restore])

  if FLAGS.model_name=='deeper_mobilenet_v1':
    #  if FLAGS.isp_checkpoint_path is not None:
        #  isp_init_func = _init_isp_func()
    variables_to_restore = [var for var in variables_to_restore if 'MobilenetV1' in var.name]
    print("Restoring isp variables and inception variables from seperate checkpoints.")

    # else:
    #   print("Restoring all variables from a single checkpoint.")
    #   pprint([var.name for var in variables_to_restore])

  inception_init_func = slim.assign_from_checkpoint_fn(
	      checkpoint_path,
	      variables_to_restore,
	  ignore_missing_vars=FLAGS.ignore_missing_vars)

  def callback(session):
      if FLAGS.model_name=='mobilenet_isp':
          if FLAGS.isp_checkpoint_path is not None:
              isp_init_func(session)
      inception_init_func(session)

  return callback

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=False,
        replica_id=0,
        num_replicas=1,
        num_ps_tasks=0)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=dataset.num_classes,
        weight_decay=FLAGS.weight_decay,
        batch_norm_decay=None,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      # Noise up the images - don't do that for models where we are preprocessing the images with an existing ISP.
      with tf.device("/cpu:0"):
        noisy_batch, a, gauss_std = sensor_model.sensor_noise_rand_light_level(images,
              [FLAGS.ll_low, FLAGS.ll_high], scale=1.0)
      bayer_mask = sensor_model.get_bayer_mask(train_image_size, train_image_size)
      inputs = noisy_batch*bayer_mask

      # These parameters are only relevant for our special ISP functions. Mobilenet for instance will just eat them and not act upon them.
      logits, end_points, _ = network_fn(images=inputs,
                                      num_classes=dataset.num_classes,
                                      alpha=a,
                                      sigma=gauss_std,
                                      bayer_mask=bayer_mask,
  				      use_anscombe=FLAGS.use_anscombe,
                                      noise_channel=FLAGS.noise_channel,
                                      num_iters=FLAGS.num_iters, num_layers=FLAGS.num_layers,
                                      isp_model_name=FLAGS.isp_model_name, is_real_data=False)


      end_points['ground_truth'] = images
      # end_points['noisy'] = noisy_batch

      #############################
      # Specify the loss function #
      #############################
      tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels,
          label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)

    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    # Add image summary for denoised image
    for end_point in end_points:
        if end_point in ['outputs', 'post_anscombe',
                         'pre_inv_anscombe']:
            summaries.add(tf.summary.image(end_point, end_points[end_point]))
        if end_point in ['mobilenet_input', 'noisy', 'inputs', 'ground_truth', 'R', 'G1', 'G2', 'B']:
            clean_image = end_points[end_point]
            summaries.add(tf.summary.image(end_point, clean_image))
            summaries.add(tf.summary.scalar('bounds/%s_min'%end_point, tf.reduce_min(clean_image)))
            summaries.add(tf.summary.scalar('bounds/%s_max'%end_point, tf.reduce_max(clean_image)))

    #################################
    # Configure the moving averages #
    #################################
    moving_average_variables = slim.get_model_variables()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    # Update ops executed locally by trainer.
    update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)

    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                      name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(train_tensor,
                        saver=saver,
                        logdir=FLAGS.train_dir,
                        master='',
                        is_chief=True,
                        init_fn=_get_init_fn(),
                        summary_op=summary_op,
                        number_of_steps=FLAGS.max_number_of_steps,
                        log_every_n_steps=FLAGS.log_every_n_steps,
                        save_summaries_secs=FLAGS.save_summaries_secs,
                        save_interval_secs=FLAGS.save_interval_secs,
                        sync_optimizer=None)


if __name__ == '__main__':
  tf.app.run()
