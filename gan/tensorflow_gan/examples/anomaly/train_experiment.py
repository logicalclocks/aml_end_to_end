# coding=utf-8
# Copyright 2019 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a GANEstimator on MNIST data using `train_and_evaluate`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow_gan.examples.anomaly import train_experiment_lib

# ML Hparams.
flags.DEFINE_integer('n_epochs', 1000, '')
flags.DEFINE_integer('data_size', 100000, '')
flags.DEFINE_integer('batch_size', 16, 'The number of images in each batch.')
flags.DEFINE_integer('noise_dims', 2, 'Dimensions of the generator noise vector')

flags.DEFINE_float('gp_weight', 10.0, 'gradient penalty weight.')

flags.DEFINE_integer('d_n_neurons', 32, '')
flags.DEFINE_integer('d_n_layers', 2, '')
flags.DEFINE_string('d_activation_fn', 'relu', '')
flags.DEFINE_bool('d_double_neurons', False, '')
flags.DEFINE_bool('d_bottlneck_neurons', False, '')
flags.DEFINE_bool('d_batch_norm', False, '')
flags.DEFINE_bool('d_batch_dropout', False, '')
flags.DEFINE_float('d_dropout_rate', 0.0, '')
flags.DEFINE_float('d_kernel_bias_reg', 0.0, '')
flags.DEFINE_float('discriminator_lr', 0.0031938, 'The discriminator learning rate.')
flags.DEFINE_float('d_l1_rate', 0.0, '')
flags.DEFINE_float('d_l2_rate', 0.0, '')

flags.DEFINE_integer('g_n_neurons', 32, '')
flags.DEFINE_integer('g_n_layers', 2, '')
flags.DEFINE_string('g_activation_fn', 'relu', '')
flags.DEFINE_bool('g_double_neurons', False, '')
flags.DEFINE_bool('g_bottlneck_neurons', False, '')
flags.DEFINE_bool('g_batch_norm', False, '')
flags.DEFINE_bool('g_batch_dropout', False, '')
flags.DEFINE_float('g_dropout_rate', 0.0, '')
flags.DEFINE_float('g_kernel_bias_reg', 0.0, '')
flags.DEFINE_float('generator_lr', 0.000076421, 'The generator learning rate.')
flags.DEFINE_float('g_l1_rate', 0.0, '')
flags.DEFINE_float('g_l2_rate', 0.0, '')

flags.DEFINE_bool('joint_train', False, 'Whether to jointly or sequentially train the generator and discriminator.')

# ML Infra.
flags.DEFINE_string('model_dir', '/tmp/tfgan_logdir/mnist-estimator-tae',
                    'Optional location to save model. If `None`, use a '
                    'default provided by tf.Estimator.')
flags.DEFINE_integer('num_train_steps', 20000, 'The maximum number of gradient steps.')
flags.DEFINE_integer('num_eval_steps', 400, 'The number of evaluation steps.')
flags.DEFINE_integer('num_reader_parallel_calls', 4, 'Number of parallel calls in the input dataset.')
flags.DEFINE_boolean('use_dummy_data', False, 'Whether to use fake data. Used for testing.')

FLAGS = flags.FLAGS



def main(_):

  hparams = train_experiment_lib.HParams(

      FLAGS.n_epochs,
      FLAGS.data_size,
      FLAGS.batch_size,
      FLAGS.noise_dims,

      FLAGS.gp_weight,

      FLAGS.d_n_neurons,
      FLAGS.d_n_layers,
      FLAGS.d_activation_fn,
      FLAGS.d_double_neurons,
      FLAGS.d_bottlneck_neurons,
      FLAGS.d_batch_norm,
      FLAGS.d_batch_dropout,
      FLAGS.d_dropout_rate,
      FLAGS.d_kernel_bias_reg,
      FLAGS.discriminator_lr,
      FLAGS.d_l1_rate,
      FLAGS.d_l2_rate,

      FLAGS.g_n_neurons,
      FLAGS.g_n_layers,
      FLAGS.g_activation_fn,
      FLAGS.g_double_neurons,
      FLAGS.g_bottlneck_neurons,
      FLAGS.g_batch_norm,
      FLAGS.g_batch_dropout,
      FLAGS.g_dropout_rate,
      FLAGS.g_kernel_bias_reg,
      FLAGS.generator_lr,
      FLAGS.g_l1_rate,
      FLAGS.g_l2_rate,

      FLAGS.joint_train,

      FLAGS.model_dir,
      FLAGS.num_train_steps,
      FLAGS.num_eval_steps,
      FLAGS.num_reader_parallel_calls,
      FLAGS.use_dummy_data

  )

  train_experiment_lib.train(hparams)


if __name__ == '__main__':
  app.run(main)
