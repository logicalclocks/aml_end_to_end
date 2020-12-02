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

import collections
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.anomaly import network_utils
from tensorflow_gan.examples.anomaly import input_utils
from tensorflow_gan.examples.anomaly import util

HParams = collections.namedtuple('HParams', [

    'n_epochs',
    'data_size',
    'batch_size',
    'noise_dims',

    'gp_weight',

    'd_n_neurons',
    'd_n_layers',
    'd_activation_fn',
    'd_double_neurons',
    'd_bottlneck_neurons',
    'd_batch_norm',
    'd_batch_dropout',
    'd_dropout_rate',
    'd_kernel_bias_reg',
    'discriminator_lr',
    'd_l1_rate',
    'd_l2_rate',

    'g_n_neurons',
    'g_n_layers',
    'g_activation_fn',
    'g_double_neurons',
    'g_bottlneck_neurons',
    'g_batch_norm',
    'g_batch_dropout',
    'g_dropout_rate',
    'g_kernel_bias_reg',
    'generator_lr',
    'g_l1_rate',
    'g_l2_rate',

    'joint_train',

    # ML Infra.
    'model_dir',
    'num_train_steps',
    'num_eval_steps',
    'num_reader_parallel_calls',
    'use_dummy_data'


])


def get_metrics(gan_model, targets):
    # TODO (davit): complete this function
    """Return metrics for MNIST experiment."""
    real_eval_crossentropy_score = util.anomaly_cross_entropy(gan_model.predictions, targets)

    # frechet_distance = util.mnist_frechet_distance(gan_model.real_data, gan_model.generated_data)
    return real_eval_crossentropy_score


def make_estimator(hparams, generator_fn, discriminator_fn):
    return tfgan.estimator.GANAnomalyEstimator(
        model_dir=hparams.model_dir,
        generator_fn=generator_fn,
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        params=hparams._asdict(),
        generator_optimizer=tf.compat.v1.train.AdamOptimizer(
            hparams.generator_lr, 0.5),
        discriminator_optimizer=tf.compat.v1.train.AdamOptimizer(
            hparams.discriminator_lr, 0.5),
        add_summaries=True,  # FIXME (davit): tfgan.estimator.SummaryType.IMAGES
        get_eval_metric_ops_fn=None,  # FIXME (davit): get_metrics
    )


def train(hparams):
    params = hparams._asdict()
    """Trains an MNIST GAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
  """

    """Input function for GANEstimator."""
    if 'batch_size' not in params:
        raise ValueError('batch_size must be in params')
    if 'data_size' not in params:
        raise ValueError('data_size must be in params')
    if 'noise_dims' not in params:
        raise ValueError('noise_dims must be in params')
    if 'n_epochs' not in params:
        raise ValueError('n_epochs must be in params')

    if 'gp_weight' not in params:
        raise ValueError('gp_weight must be in params')

    if 'd_n_neurons' not in params:
        raise ValueError('d_n_neurons must be in params')
    if 'd_n_layers' not in params:
        raise ValueError('d_n_layers must be in params')
    if 'd_activation_fn' not in params:
        raise ValueError('d_activation_fn must be in params')
    if 'd_double_neurons' not in params:
        raise ValueError('d_double_neurons must be in params')
    if 'd_bottlneck_neurons' not in params:
        raise ValueError('d_bottlneck_neurons must be in params')
    if 'd_batch_norm' not in params:
        raise ValueError('d_batch_norm must be in params')
    if 'd_batch_dropout' not in params:
        raise ValueError('d_batch_dropout must be in params')
    if 'd_dropout_rate' not in params:
        raise ValueError('d_dropout_rate must be in params')
    if 'd_kernel_bias_reg' not in params:
        raise ValueError('d_kernel_bias_reg must be in params')
    if 'discriminator_lr' not in params:
        raise ValueError('discriminator_lr must be in params')
    if 'd_l1_rate' not in params:
        raise ValueError('d_l1_rate must be in params')
    if 'd_l2_rate' not in params:
        raise ValueError('d_l2_rate must be in params')

    if 'g_n_neurons' not in  params:
        raise ValueError('g_n_neurons must be in params')
    if 'g_n_layers' not in params:
        raise ValueError('g_n_layers must be in params')
    if 'g_activation_fn' not in params:
        raise ValueError('g_activation_fn must be in params')
    if 'g_double_neurons' not in params:
        raise ValueError('g_double_neurons must be in params')
    if 'g_bottlneck_neurons' not in params:
        raise ValueError('g_bottlneck_neurons must be in params')
    if 'g_batch_norm' not in params:
        raise ValueError('g_batch_norm must be in params')
    if 'g_batch_dropout' not in params:
        raise ValueError('g_batch_dropout must be in params')
    if 'g_dropout_rate' not in params:
        raise ValueError('g_dropout_rate must be in params')
    if 'g_kernel_bias_reg' not in params:
        raise ValueError('g_kernel_bias_reg must be in params')
    if 'generator_lr' not in params:
        raise ValueError('generator_lr must be in params')
    if 'g_l1_rate' not in params:
        raise ValueError('g_l1_rate must be in params')
    if 'g_l2_rate' not in params:
        raise ValueError('g_l2_rate must be in params')

    if 'model_dir' not in params:
        raise ValueError('model_dir must be in params')


    bs = params['batch_size']
    ds = params['data_size']
    nd = params['noise_dims']
    ne = params['n_epochs']
    train_data = input_utils.norm_gen(input_dim=nd, n_samples=ds)
    eval_data = input_utils.norm_ano_gen(input_dim=nd, n_samples=ds)
    train_input_fn = input_utils.InputFunction(train_data[0], train_data[1], bs, nd, ne, tf.estimator.ModeKeys.TRAIN)
    eval_input_fn = input_utils.InputFunction(eval_data[0], eval_data[1], bs, nd, ne, tf.estimator.ModeKeys.EVAL)

    pred_input_fn = input_utils.InputFunction(train_data[0], train_data[1], bs, nd, ne, tf.estimator.ModeKeys.PREDICT)

    # TODO (davit): here construct final version of gen and dis models with hparams or as input
    # TODO (davit): add extra hparams to construct model
    def _unconditional_generator(noise, mode, n_neurons=hparams.g_n_neurons, n_layers=hparams.g_n_layers,
                                 output_dim=hparams.noise_dims,
                                 activation_fn=hparams.g_activation_fn, double_neurons=hparams.g_double_neurons,
                                 bottleneck_neurons=hparams.g_bottlneck_neurons, batch_norm=hparams.g_batch_norm,
                                 batch_dropout=hparams.g_batch_dropout, dropout_rate=hparams.g_dropout_rate,
                                 kernel_bias_reg=hparams.g_kernel_bias_reg, l1_rate=hparams.g_l1_rate,
                                 l2_rate=hparams.g_l2_rate):
        """... generator with extra argument for tf.Estimator's `mode`."""
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        return network_utils.unconditional_generator(noise=noise, n_neurons=n_neurons, n_layers=n_layers,
                                                     output_dim=output_dim, activation_fn=activation_fn,
                                                     double_neurons=double_neurons,
                                                     bottleneck_neurons=bottleneck_neurons,
                                                     batch_norm=batch_norm, batch_dropout=batch_dropout,
                                                     dropout_rate=dropout_rate, kernel_bias_reg=kernel_bias_reg,
                                                     l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)

    # TODO (davit): add extra hparams to construct model
    def _unconditional_discriminator(inputs, mode, n_neurons=hparams.d_n_neurons, n_layers=hparams.d_n_layers,
                                     output_dim=hparams.noise_dims,
                                     activation_fn=hparams.d_activation_fn, double_neurons=hparams.d_double_neurons,
                                     bottleneck_neurons=hparams.d_bottlneck_neurons, batch_norm=hparams.d_batch_norm,
                                     batch_dropout=hparams.d_batch_dropout, dropout_rate=hparams.d_dropout_rate,
                                     kernel_bias_reg=hparams.d_kernel_bias_reg, l1_rate=hparams.d_l1_rate,
                                     l2_rate=hparams.d_l2_rate):
        """... discriminator with extra argument for tf.Estimator's `mode`."""
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        return network_utils.unconditional_discriminator(input=inputs, n_neurons=n_neurons, n_layers=n_layers,
                                                         output_dim=output_dim,
                                                         activation_fn=activation_fn, double_neurons=double_neurons,
                                                         bottleneck_neurons=bottleneck_neurons, batch_norm=batch_norm,
                                                         batch_dropout=batch_dropout, dropout_rate=dropout_rate,
                                                         kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate,
                                                         l2_rate=l2_rate,
                                                         is_training=is_training)

    estimator = make_estimator(hparams=hparams, generator_fn=_unconditional_generator,
                               discriminator_fn=_unconditional_discriminator)

    # TODO (davit): round up ds here
    gen_update_hook = input_utils.UpdateGeneratorInputsHook(estimator, pred_input_fn, ds, bs, nd)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=hparams.num_train_steps, hooks=[train_input_fn.init_hook, gen_update_hook])
    eval_spec = tf.estimator.EvalSpec(
        name='default', input_fn=eval_input_fn, steps=hparams.num_eval_steps, hooks=[eval_input_fn.init_hook])

    # Run training and evaluation for some steps.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    eval_result = estimator.evaluate(input_fn=eval_input_fn, steps=hparams.num_eval_steps)

    print(eval_result)
    return eval_result

    # FIXME (davit): here labels are present so make sure they are None
    # Generate predictions and write them to disk.
    # yields_prediction = estimator.predict(input_fn)
    # predictions = np.array([next(yields_prediction) for _ in xrange(100)])
    # write_predictions_to_disk(predictions, hparams.model_dir, hparams.num_train_steps)
