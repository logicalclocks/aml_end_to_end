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

from tensorflow_gan.examples.anomaly import train_gan_enc_experiment_lib

# ML Hparams.
flags.DEFINE_string('model_name', 'gan_enc_anomaly', '')
flags.DEFINE_integer('n_epochs', 10, '')
flags.DEFINE_integer('data_size', 1000, '')
flags.DEFINE_integer('batch_size', 16, 'The number of images in each batch.')
flags.DEFINE_integer('noise_dims', 64, 'Dimensions of the generator noise vector, this should =encoder_start_num_neurons/encoder_n_layers')
flags.DEFINE_integer('g_output_dim', 365, 'Dimensions of the generator output')
flags.DEFINE_integer('d_output_dim', 1, 'Dimensions of the discriminator logits')
flags.DEFINE_integer('feature_dim', 1, '')
flags.DEFINE_integer('time_steps', 365, '')
flags.DEFINE_bool('timeseries', False, '')

flags.DEFINE_float('gp_weight', 1.0, 'gradient penalty weight.')

flags.DEFINE_integer('d_n_neurons', 256, '')
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

flags.DEFINE_integer('g_n_neurons', 256, '')
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

flags.DEFINE_integer('encoder_start_num_neurons', 128, '')
flags.DEFINE_integer('encoder_n_layers', 2, '')

flags.DEFINE_bool('joint_train', False, 'Whether to jointly or sequentially train the generator and discriminator.')

flags.DEFINE_string('experiment_type', 'hp', '')

flags.DEFINE_integer('num_train_steps', 100, 'The maximum number of gradient steps.')
flags.DEFINE_integer('num_eval_steps', 40, 'The number of evaluation steps.')
flags.DEFINE_integer('num_summary_steps', 10, '')
flags.DEFINE_integer('log_step_count_steps', 10, '')
flags.DEFINE_integer('save_checkpoints_steps', 10, '')

flags.DEFINE_string('train_data', './datasets/demo.tfrecord', '')
flags.DEFINE_string('eval_data', './datasets/demo.tfrecord', '')
flags.DEFINE_string('pred_data', './datasets/demo.tfrecord', '')

flags.DEFINE_string('model_dir', '/tmp/tfgan_logdir/mnist-estimator-tae',
                    'Optional location to save model. If `None`, use a '
                    'default provided by tf.Estimator.')
flags.DEFINE_string('simple_profiler_output_dir', '/tmp/tfgan_logdir/simple_profiler_output_dir', '')
flags.DEFINE_string('context_profiler_output_dir', '/tmp/tfgan_logdir/context_profiler_output_dir', '')

# ML Infra.
flags.DEFINE_integer('num_gpus_per_worker', 1, '')
flags.DEFINE_integer('num_reader_parallel_calls', 4, 'Number of parallel calls in the input dataset.')
flags.DEFINE_boolean('use_dummy_data', False, 'Whether to use fake data. Used for testing.')

FLAGS = flags.FLAGS

def main(_):

  hparams = train_gan_enc_experiment_lib.HParams(

  FLAGS.model_name,

  FLAGS.n_epochs,
  FLAGS.data_size,
  FLAGS.batch_size,
  FLAGS.noise_dims,
  FLAGS.g_output_dim,
  FLAGS.d_output_dim,
  FLAGS.feature_dim,
  FLAGS.time_steps,
  FLAGS.timeseries,

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

  FLAGS.encoder_start_num_neurons,
  FLAGS.encoder_n_layers,

  FLAGS.joint_train,

  FLAGS.experiment_type,

  FLAGS.num_train_steps,
  FLAGS.num_eval_steps,
  FLAGS.num_summary_steps,
  FLAGS.log_step_count_steps,
  FLAGS.save_checkpoints_steps,

  FLAGS.train_data,
  FLAGS.eval_data,
  FLAGS.pred_data,

  FLAGS.model_dir,
  FLAGS.simple_profiler_output_dir,
  FLAGS.context_profiler_output_dir,
  FLAGS.num_gpus_per_worker,
  FLAGS.num_reader_parallel_calls,
  FLAGS.use_dummy_data

  )

  train_gan_enc_experiment_lib.train(hparams)


if __name__ == '__main__':
  app.run(main)
