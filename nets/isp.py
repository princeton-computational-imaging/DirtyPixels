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
"""Contains the definition of the Inception V4 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception_utils

slim = tf.contrib.slim

def isp_arg_scope(weight_decay=0.00004,
                  use_batch_norm=True,
                  batch_norm_decay=0.95,
                  batch_norm_epsilon=0.0001):
   """Defines the default arg scope for inception models.

   Args:
     weight_decay: The weight decay to use for regularizing the model.
     use_batch_norm: "If `True`, batch_norm is applied after each convolution.
     batch_norm_decay: Decay for batch norm moving average.
     batch_norm_epsilon: Small float added to variance to avoid dividing by zero
       in batch norm.

   Returns:
     An `arg_scope` to use for the inception models.
   """
   print("weight decay = ", weight_decay)
   print("batch norm decay = ", batch_norm_decay)
   batch_norm_params = {
       # Decay for the moving averages.
       'decay': batch_norm_decay,
       # epsilon to prevent 0s in variance.
       'epsilon': batch_norm_epsilon,
       # collection containing update_ops.
       'updates_collections': tf.GraphKeys.UPDATE_OPS,
       'center': True,
       'scale': False,
   }
   if use_batch_norm:
     normalizer_fn = slim.batch_norm
     normalizer_params = batch_norm_params
   else:
     normalizer_fn = None
     normalizer_params = {}
   # Set weight_decay for weights in Conv and FC layers.
   with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       weights_regularizer=slim.l2_regularizer(weight_decay)):
     with slim.arg_scope(
         [slim.conv2d],
         weights_initializer=slim.variance_scaling_initializer(),
         activation_fn=tf.nn.relu,
         normalizer_fn=normalizer_fn,
         normalizer_params=normalizer_params) as sc:
       return sc

def anscombe(data, sigma, alpha, scale=255.0, is_real_data=False):
    """Transform N(mu,sigma^2) + \alpha Pois(y) into N(0,scale^2) noise."""
    if is_real_data:
      z = data/alpha[:,None,None,:]
      sigma_hat = sigma/alpha
      sqrt_term = z + 3./8. + tf.square(sigma_hat)[:,None,None,:]
    else:
      z = data/alpha[:,None,None,None]
      sigma_hat = sigma/alpha
      sqrt_term = z + 3./8. + tf.square(sigma_hat)[:,None,None,None]
    
    sqrt_term = tf.maximum(sqrt_term, 0.0)

    return 2*tf.sqrt(sqrt_term)


def inv_anscombe(data, sigma, alpha, scale=1.0, unbiased=False, is_real_data=False):
    """Invert anscombe transform."""
    sigma_hat = sigma/alpha
    if is_real_data:
      z = .25* tf.square(data) - 1./8 - tf.square(sigma_hat)[:,None,None,:]
      if unbiased:
        z = z + .25*tf.sqrt(3./2)*data**-1 - 11./8.*data**-2 + 5./8.*tf.sqrt(3./2)*data**-3
      result = z*alpha[:,None,None,:]
    else:
      z = .25* tf.square(data) - 1./8 - tf.square(sigma_hat)[:,None,None,None]
      #data = tf.Print(data, ["data", tf.reduce_max(data), tf.reduce_min(data)])
      
      #z = tf.maximum(z, 0)
      if unbiased:
        z = z + .25*tf.sqrt(3./2)*data**-1 - 11./8.*data**-2 + 5./8.*tf.sqrt(3./2)*data**-3
      result = z*alpha[:,None,None,None]
    return result
    #return tf.clip_by_value(result, 0.0, scale)

def prox_grad_isp(inputs,
                  alpha,
                  sigma,
                  bayer_mask,
                  num_iters=4,
                  num_channels=3,
                  num_layers=5,
                  kernel=None,
                  num_classes=1001,
                  is_training=True,
                  scale=1.0,
                  use_anscombe=True,
                  noise_channel=True, 
                  use_chen_unet=False, 
                  is_real_data=True):

    end_points = {}
    end_points['inputs'] = inputs
    if use_anscombe and alpha is not None:
        print(("USING THE ANCOMB TRANSFORM with scale %f" % scale) + "!"*10)
        true_img = anscombe(inputs, alpha=alpha, sigma=sigma, scale=scale, is_real_data=is_real_data)
        min_offset = tf.reduce_min(true_img, [1,2,3], keep_dims=True)
        max_scale = tf.reduce_max(true_img, [1,2,3], keep_dims=True)
        noise_scale = scale/(max_scale - min_offset)
        true_img = (true_img - min_offset)*noise_scale
        noise_ch = noise_scale
        end_points['post_anscombe'] = true_img
    else:
        true_img = inputs
        noise_ch = sigma[:,None,None,None]

    if not noise_channel:
        noise_ch = None
    else:
        print(("USING NOISE CHANNEL"))
        dims = [d.value for d in inputs.get_shape()]
        noise_ch = tf.tile(noise_ch, [1, dims[1], dims[2], 1])

    if use_chen_unet:
      print('USING UNET AS ISP (NON-PROX GRAD)')
      from nets import unet
      ans_x_out = unet.unet(true_img)
      end_points = {}

    else:
      ans_x_out, end_points = prox_grad(true_img, bayer_mask, end_points, num_layers=num_layers,
                          num_iters=num_iters, noise_channel=noise_ch, is_training=is_training)
    # ans_x_out, end_points = prox_grad(true_img, bayer_mask, end_points, num_layers=num_layers,
    #                       num_iters=num_iters, noise_channel=noise_ch, is_training=is_training)

    if use_anscombe and alpha is not None:
        end_points['pre_inv_anscombe'] = ans_x_out
        ans_x_out = ans_x_out/noise_scale + min_offset
        ans_x_out = inv_anscombe(ans_x_out, alpha=alpha, sigma=sigma, scale=scale, is_real_data=is_real_data)
    end_points['outputs'] = ans_x_out
    return ans_x_out, end_points


def prox_grad(inputs, bayer_mask, end_points, num_layers=5, num_iters=4, lambda_init=1.0,
        is_training=True, scope='gauss_den', noise_channel=None):
  flat_inputs = tf.reduce_sum(inputs, 3, keep_dims=True)
  with tf.variable_scope(scope, 'gauss_den', [inputs]) as sc:
    xk = inputs
    lam = slim.variable(name='lambda', shape=[], initializer=tf.constant_initializer(lambda_init))
    end_points['lambda'] = lam
    beta_init = 1.0
    for t in range(num_iters):
      with tf.variable_scope('iter_%i'% t):
         with slim.arg_scope([slim.batch_norm, slim.dropout],
                              is_training=is_training):
             # Collect outputs for conv2d, fully_connected and max_pool2d.
             beta_init *=  2.0 # Continuation scheme as proposed in http://www.caam.rice.edu/~yzhang/reports/tr0710_rev.pdf, algorithm 2
             beta = slim.variable(name='beta', shape=[], initializer=tf.constant_initializer(beta_init))
             end_points['beta%s'%t] = beta
             with tf.variable_scope('prior_grad') as prior_scope:
                #curr_z = cnn_proximal(xk, num_layers, 3, noise_channel, width=12, rate=1)
                if noise_channel is None:
                    concat_xk = xk
                else:
                    concat_xk = tf.concat([xk, noise_channel], 3)
                curr_z = unet_res(concat_xk, 0, 'unet')
             #end_points['prior_grad_%i' % t] = curr_z
             tmp = xk - curr_z
             xk = (lam*bayer_mask*inputs + beta*tmp)/(lam*bayer_mask + beta)
             #end_points['iter_%i' % t] = xk

  return xk, end_points

def unet_res(inputs, depth, scope, max_depth=2):
   # U-NET operating at a given resolution.
   shape = [d.value for d in inputs.get_shape()]
   print(depth, shape)
   ch = max(shape[3]*2, 8)
   with tf.variable_scope('depth_%s' % depth, values=[inputs]) as scope:
        if depth == 0:
          outputs = slim.conv2d(inputs, ch, [3, 3], rate=2, scope='conv_in', normalizer_fn=None)
        else:
          outputs = slim.conv2d(inputs, ch, [3, 3], scope='conv_in')
        outputs = slim.conv2d(outputs, ch, [3, 3], scope='conv_1')
        downsamp = slim.avg_pool2d(outputs, [2, 2])
   if depth < max_depth:
     lower = unet_res(downsamp, depth+1, scope, max_depth)
     outputs = tf.concat([outputs, lower], 3)
   with tf.variable_scope('depth_%s' % depth, values=[outputs]) as scope:
        outputs = slim.conv2d(outputs, ch, [3, 3], scope='conv_2')
        if depth > 0:
           outputs = slim.conv2d(outputs, ch, [3, 3], scope='out_conv')
           outputs = slim.conv2d_transpose(outputs, ch//2, [2,2], stride=2, scope='up_conv',
               activation_fn=None, normalizer_fn=None)
        else:
           outputs = slim.conv2d(outputs, 3, [3, 3], scope='out_conv',
             activation_fn=None, normalizer_fn=None)
   return outputs
