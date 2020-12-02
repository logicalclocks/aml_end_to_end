# coding=utf-8
# Copyright 2019 The Swedbank GAN Anomaly detection Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _add_hidden_layer_summary(value):
    # TODO (davit): write docstring
    ##################################################################################
    # TODO (davit): come up with better idea how to handle null values (e.g. implement yearly stop)
    value = tf.where(tf.compat.v1.is_nan(value), tf.compat.v1.zeros_like(value), value)
    ##################################################################################
    tf.compat.v1.summary.scalar('fraction_of_zero_values', tf.compat.v1.nn.zero_fraction(value))
    tf.compat.v1.summary.histogram('activation', value)


def _fullylayer(x, units, activation_fn, name, batch_norm=False, batch_dropout=False, dropout_rate=0.0,
                kernel_bias_reg=False, l1_rate=0.0, l2_rate=0.0, is_training=False):
    # TODO (davit): add docstring here

    if activation_fn == 'relu':
        activation_fn = tf.nn.relu
    if activation_fn == 'leaky_relu':
        activation_fn = tf.nn.leaky_relu
    if activation_fn == 'tanh':
        activation_fn = tf.nn.tanh
    if activation_fn == 'selu':
        activation_fn = tf.nn.selu
    if activation_fn == 'linear':
        activation_fn = None

    # Add regularizers when creating variables or layers:
    if kernel_bias_reg:
        dence_layer = tf.compat.v1.layers.dense(inputs=x, units=units, activation=activation_fn,
                                                kernel_regularizer=tf.keras.regularizers.l1(l=l1_rate),
                                                bias_regularizer=tf.keras.regularizers.l2(l=l2_rate), name=name)
    else:
        dence_layer = tf.compat.v1.layers.dense(inputs=x, units=units, activation=activation_fn, name=name)
    if batch_dropout:
        dence_layer = tf.compat.v1.layers.dropout(dence_layer, rate=dropout_rate, training=is_training, name=name)
    if batch_norm:
        dence_layer = tf.compat.v1.layers.batch_normalization(dence_layer, training=is_training, momentum=0.95,
                                                              name=name)

    return dence_layer


def _create_minibatch_feature_layer(x, units, activation_fn, name, num_kernel=5, kernel_dim=3, batch_norm=False,
                                    batch_dropout=False, dropout_rate=0.5, kernel_bias_reg=False, l1_rate=0.0005,
                                    l2_rate=0.9995, is_training=False):
    # TODO (davit): add docstring here

    """
    In the paper "Improved Techniques for Training GANs" (https://arxiv.org/abs/1606.03498), the authors present a
    technique to avoid mode-collapse while training called minibatch-discrimination (section 3.2). This method is
    implemented here - concatenating the differences of the previous layer in the batch to the current layer input.
    :param x: layer's input.
    :param num_kernel:
    :param kernel_dim:
    :param scope_name: Name to use in tf.name_scope
    :return: hidden layer - the concatenation of the input and the minibatch features calculated.
    """
    h = _fullylayer(x, units, activation_fn, name, batch_norm=False, batch_dropout=False, dropout_rate=0.5,
                    kernel_bias_reg=False, l1_rate=0.0005, l2_rate=0.9995, is_training=False)

    h = tf.reshape(h, shape=[-1, num_kernel, kernel_dim])
    diffs = tf.compat.v1.expand_dims(h, axis=3) - tf.compat.v1.expand_dims(tf.compat.v1.transpose(h, perm=[1, 2, 0]),
                                                                           axis=0)
    diffs = tf.compat.v1.abs(diffs)
    diff = tf.compat.v1.reduce_sum(diffs, axis=2)
    diff = tf.compat.v1.exp(-diff)
    minibatch_features = tf.compat.v1.reduce_sum(diff, axis=2)
    return tf.compat.v1.concat(values=[x, minibatch_features], axis=1)


def _dense_layers(x, n_neurons, n_layers, name_scope, minibatch_size, activation_fn=None, double_neurons=False,
                  bottleneck_neurons=False, batch_norm=True, batch_dropout=False, dropout_rate=0.5,
                  kernel_bias_reg=False, l1_rate=0.0005, l2_rate=0.9995, is_training=False):
    # activation: Activation function (callable). Set it to None to maintain a linear activation.

    # n_layers must be at least 1
    if n_layers <= 0:
        raise ValueError("n_layers must be at least 1! but at the moment it is" + str(n_layers))
    # n_neurons must be at least 1
    if n_neurons <= 0:
        raise ValueError("n_neurons must be at least 1! but at the moment it is" + str(n_neurons))

    if double_neurons and bottleneck_neurons:
        raise ValueError("double_neurons and bottleneck_neurons can't both true")

    with tf.compat.v1.variable_scope(name_scope, [x]):

        with tf.compat.v1.variable_scope('layer_{}'.format(1), values=(x,)):
            fc = _fullylayer(x, units=n_neurons, activation_fn=activation_fn, name='fc%i' % 1, batch_norm=batch_norm,
                             batch_dropout=batch_dropout, dropout_rate=dropout_rate, kernel_bias_reg=kernel_bias_reg,
                             l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)
            _add_hidden_layer_summary(fc)
        if n_layers > 1:
            for layer in range(2, n_layers + 1):
                with tf.compat.v1.variable_scope('layer_{}'.format(layer), values=(fc,)):
                    if double_neurons:
                        n_neurons = int(n_neurons * 2)
                    elif bottleneck_neurons:
                        n_neurons = int(n_neurons / 2)
                    # Minibatch features layer or regular hidden layer:
                    if minibatch_size > 1:
                        fc = _create_minibatch_feature_layer(fc, units=n_neurons, activation_fn=activation_fn,
                                                             name='fc%i' % layer, batch_norm=batch_norm,
                                                             batch_dropout=batch_dropout, dropout_rate=dropout_rate,
                                                             kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate,
                                                             l2_rate=l2_rate, is_training=is_training)
                    else:
                        fc = _fullylayer(fc, units=n_neurons, activation_fn=activation_fn, name='fc%i' % layer,
                                         batch_norm=batch_norm, batch_dropout=batch_dropout, dropout_rate=dropout_rate,
                                         kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate, l2_rate=l2_rate,
                                         is_training=is_training)
                _add_hidden_layer_summary(fc)

    return fc


def generator_helper(noise, n_neurons, n_layers, output_dim, activation_fn, double_neurons, bottleneck_neurons,
                     batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate, is_training):
    # TODO (davit): edit docstring
    """Core ... generator.

    This function is reused between the different GAN anomaly models (...).

    Args:
      noise: A 2D Tensor of shape [batch size, noise dim].

      l2_rate: weight_decay: The value of the l2 weight decay.
      is_training: If `True`, batch norm uses batch statistics. If `False`, batch
        norm uses the exponential moving average collected from population
        statistics.

    Returns:
      A generated image in the range [-1, 1] if its relu
    """

    # FIXME (afrooz): 2nd layer n_neurons should follow more clean evolotion (e.g. doesn't make sence to jump from 8 to 4096)
    G_ff_dense_net = _dense_layers(noise,
                                   n_neurons,
                                   n_layers,
                                   name_scope='G_ff_dense_net',
                                   minibatch_size=0,
                                   activation_fn=activation_fn,
                                   double_neurons=double_neurons,
                                   bottleneck_neurons=bottleneck_neurons, #TODO (davit): true doesn't seem reasonable here as it needs to be upscaled from noise
                                   batch_norm=batch_norm,
                                   batch_dropout=batch_dropout,
                                   dropout_rate=dropout_rate,
                                   kernel_bias_reg=kernel_bias_reg,
                                   l1_rate=l1_rate,
                                   l2_rate=l2_rate,
                                   is_training=is_training)

    # TODO (afrooz): this is just FYI. Please note that here it forces to output correct dimention
    g_output_logit = tf.compat.v1.layers.dense(G_ff_dense_net, output_dim)

    g_output_logit = tf.compat.v1.identity(g_output_logit, name="g_output_logit")
    tf.compat.v1.summary.histogram('g_output_logit', g_output_logit)

    # TODO (davit): what is the activation function here if we use different ones in hp?
    #  tf.nn.relu upper layers and tf.tanh as last ??

    return g_output_logit


def discriminator_helper(input, n_neurons, n_layers, activation_fn, double_neurons, bottleneck_neurons,
                         batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate, is_training):
    # TODO (davit): edit docstring
    """Core ... discriminator.

    This function is reused between the different GAN modes (unconditional,
    conditional, etc).

    Args:
      input: Real or generated MNIST digits. Should be in the range [-1, 1].
      l2_rate: weight_decay: The L2 weight decay.

    Returns:
      Final fully connected discriminator layer. [batch_size, ...].
    """

    # TODO (davit): what is the activation function here?
    #  leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)

    D_ff_dense_net = _dense_layers(input,
                                   n_neurons,
                                   n_layers,
                                   name_scope='D_ff_dense_net',
                                   minibatch_size=1,
                                   activation_fn=activation_fn,
                                   double_neurons=double_neurons,
                                   bottleneck_neurons=bottleneck_neurons,
                                   batch_norm=batch_norm,
                                   batch_dropout=batch_dropout,
                                   dropout_rate=dropout_rate,
                                   kernel_bias_reg=kernel_bias_reg,
                                   l1_rate=l1_rate,
                                   l2_rate=l2_rate,
                                   is_training=is_training)

    return D_ff_dense_net


def unconditional_generator(noise, n_neurons, n_layers, output_dim, activation_fn, double_neurons, bottleneck_neurons,
                            batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate,
                            is_training=True):
    # TODO (davit): revise docstring
    """Generator to produce unconditional ... images.

  Args:
    noise: A single Tensor representing noise.
    weight_decay: The value of the l2 weight decay.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.

  Returns:
    A generated image in the range [-1, 1].
  """
    # with tf.compat.v1.variable_scope(scope):  # TODO (davit): is this necessaery? reuse=reuse

    return generator_helper(noise, n_neurons, n_layers, output_dim, activation_fn, double_neurons, bottleneck_neurons,
                            batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate, is_training)


def unconditional_discriminator(input, n_neurons, n_layers, output_dim, activation_fn, double_neurons,
                                bottleneck_neurons, batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate,
                                l2_rate, is_training=True):
    # TODO (davit): revise docstring
    """Discriminator network on unconditional ... digits.

  Args:
    img: Real or generated MNIST digits. Should be in the range [-1, 1].
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    weight_decay: The L2 weight decay.

  Returns:
    Logits for the probability that the image is real.
  """

    # with tf.compat.v1.variable_scope(scope):  # TODO (davit): is this necessary? reuse=reuse
    D_ff_dense_net = discriminator_helper(input,
                                          n_neurons,
                                          n_layers,
                                          activation_fn=activation_fn,
                                          double_neurons=double_neurons,
                                          bottleneck_neurons=bottleneck_neurons,
                                          batch_norm=batch_norm,
                                          batch_dropout=batch_dropout,
                                          dropout_rate=dropout_rate,
                                          kernel_bias_reg=kernel_bias_reg,
                                          l1_rate=l1_rate,
                                          l2_rate=l2_rate,
                                          is_training=is_training)

    d_output_logit = tf.compat.v1.layers.dense(D_ff_dense_net, output_dim,
                                               kernel_regularizer=tf.keras.regularizers.l2(l=l2_rate))

    # TODO (davit): what is the activation function here if we use different ones in hp?
    predictions = tf.nn.sigmoid(d_output_logit)

    return d_output_logit, predictions


def make_encoder(inputs, start_num_neurons, n_layers, is_training, rand_sampling='normal', z_reg_type=None):
    # TODO (davit): write docstring
    """

    see: f-AnoGan paper for more details

    :param inputs:
    :param start_num_neurons:
    :param n_layers:
    :param is_training:
    :param rand_sampling:
    :param z_reg_type:
    :return:
    """

    # TODO (davit): everything here needs to be hyperparameter
    output = _dense_layers(x=inputs, n_neurons=start_num_neurons, n_layers=n_layers, name_scope="Encoder",
                           minibatch_size=0, activation_fn=None, double_neurons=False, bottleneck_neurons=True,
                           batch_norm=False, batch_dropout=False, dropout_rate=0.5, kernel_bias_reg=False,
                           l1_rate=0.0005, l2_rate=0.9995, is_training=is_training)

    if z_reg_type is None:
        return output
    elif z_reg_type == 'tanh_fc':
        return tf.nn.tanh(output)
    elif z_reg_type == '3s_tanh_fc':
        return tf.nn.tanh(output) * 3
    elif z_reg_type == '05s_tanh_fc':
        return tf.nn.tanh(output) * 0.5
    elif z_reg_type == 'hard_clip':
        return tf.clip_by_value(output, -1., 1.)
    elif z_reg_type == '3s_hard_clip':
        return tf.clip_by_value(output, -3., 3.)
    elif z_reg_type == '05s_hard_clip':
        return tf.clip_by_value(output, -0.5, 0.5)
    elif z_reg_type == 'stoch_clip':  ## IMPLEMENTS STOCHASTIC CLIPPING  -->> https://arxiv.org/pdf/1702.04782.pdf
        if rand_sampling == 'unif':
            condition = tf.greater(tf.abs(output), 1.)
            true_case = tf.random_uniform(output.get_shape(), minval=-1., maxval=1.)
        elif rand_sampling == 'normal':
            condition = tf.greater(tf.abs(output), 3.)
            true_case = tf.random_normal(output.get_shape())
        return tf.where(condition, true_case, output)


#######################################################################################################################
def _d1_cnn(x, filters, kernel_size, name, stride, padding, activation_fn,
            batch_norm, batch_dropout, dropout_rate, kernel_bias_reg,
            l1_rate, l2_rate, is_training):
    # Add regularizers when creating variables or layers:
    if kernel_bias_reg:
        l1_reg = tf.keras.regularizers.l1(l=l1_rate)
        l2_reg = tf.keras.regularizers.l2(l=l2_rate)
    else:
        l1_reg = tf.keras.regularizers.l1(l=l1_rate)
        l2_reg = tf.keras.regularizers.l2(l=l2_rate)

    # conv_net = ttf.compat.v1.layers.conv1d(x, filters=filters, kernel_size=kernel_size, strides=stride,
    #                             padding=padding,
    #                             data_format='channels_last',
    #                             dilation_rate=1,
    #                             activation=activation_fn,
    #                             use_bias=True,
    #                             kernel_initializer=None,
    #                             bias_initializer=tf.zeros_initializer(),
    #                             kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=l1_rate,
    #                                                                                 scope="%s_l1_regularizer" % name),
    #                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_rate,
    #                                                                               scope="%s_l2_regularizer" % name),
    #                             activity_regularizer=None,
    #                             kernel_constraint=None,
    #                             bias_constraint=None,
    #                             trainable=True,
    #                             name=name,
    #                             reuse=None
    #                             )

    conv_net = tf.keras.layers.Conv1D(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=stride,
                                      padding=padding,
                                      data_format='channels_last',
                                      dilation_rate=1,
                                      activation=activation_fn,
                                      use_bias=True,
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros',
                                      kernel_regularizer=l1_reg,
                                      bias_regularizer=l2_reg,
                                      activity_regularizer=None,
                                      kernel_constraint=None,
                                      bias_constraint=None
                                      )(x)


    if batch_dropout:
        # conv_net = tf.layers.dropout(conv_net, rate=dropout_rate, training=is_training, name=name)
        conv_net = tf.keras.layers.Dropout(rate=dropout_rate, training=is_training, noise_shape=None, seed=None)(
            conv_net)

    if batch_norm:
        # conv_net = tf.compat.v1.layers.batch_normalization(conv_net, training=is_training, momentum=0.95, name=name)
        conv_net = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
            fused=None, trainable=is_training, virtual_batch_size=None, adjustment=None, name=name,
        )(conv_net)

    return conv_net


def _conv1d(x, n_layers, filters, kernel_size, name, pool_size, stride, padding='same',
           activation_fn=tf.nn.relu, batch_norm=False, batch_dropout=False, dropout_rate=0.0, kernel_bias_reg=False,
           l1_rate=0.0, l2_rate=0.0, is_training=False):

    if n_layers == 1:
        with tf.compat.v1.variable_scope('conv_layer_{}'.format(1), values=(x,)):
            # conv_net = tf.layers.conv1d(x, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)
            conv_net = _d1_cnn(x, filters=filters, kernel_size=kernel_size, name=name,
                               stride=stride, padding=padding, activation_fn=activation_fn, batch_norm=batch_norm,
                               batch_dropout=batch_dropout, dropout_rate=dropout_rate, kernel_bias_reg=kernel_bias_reg,
                               l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)

            if pool_size > 1:
                # conv_net = tf.layers.max_pooling1d(conv_net, pool_size, strides=stride, padding='SAME', name=name)
                conv_net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding=padding)(conv_net)
    else:
        with tf.compat.v1.variable_scope('layer_{}'.format(1), values=(x,)):
            #            conv_net = tf.layers.conv1d(x, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)
            conv_net = _d1_cnn(x, filters=filters, kernel_size=kernel_size, name=name,
                               stride=stride, padding=padding, activation_fn=activation_fn, batch_norm=batch_norm,
                               batch_dropout=batch_dropout, dropout_rate=dropout_rate, kernel_bias_reg=kernel_bias_reg,
                               l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)
            if pool_size > 1:
                # conv_net = tf.layers.max_pooling1d(conv_net, pool_size, strides=stride, padding='SAME', name=name)
                conv_net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding=padding)(conv_net)
            for layer in range(2, n_layers + 1):
                with tf.compat.v1.variable_scope('layer_{}'.format(layer), values=(conv_net,)):
                    #                    conv_net = tf.layers.conv1d(conv_net, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)
                    conv_net = _d1_cnn(conv_net, filters=filters, kernel_size=kernel_size, name=name,
                                       stride=stride, padding=padding, activation_fn=activation_fn,
                                       batch_norm=batch_norm,
                                       batch_dropout=batch_dropout, dropout_rate=dropout_rate,
                                       kernel_bias_reg=kernel_bias_reg,
                                       l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)
                    if pool_size > 1:
                        # conv_net = tf.layers.max_pooling1d(conv_net, pool_size, strides=stride, padding='SAME', name=name)
                        conv_net = tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding=padding)(conv_net)
    return conv_net


def _deconv1d(x, n_layers, filters, kernel_size, name, up_size, stride, padding='same',
           activation_fn=tf.nn.relu, batch_norm=False, batch_dropout=False, dropout_rate=0.0, kernel_bias_reg=False,
           l1_rate=0.0, l2_rate=0.0, is_training=False):

    if n_layers == 1:
        with tf.compat.v1.variable_scope('conv_layer_{}'.format(1), values=(x,)):
            # conv_net = tf.layers.conv1d(x, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)
            conv_net = _d1_cnn(x, filters=filters, kernel_size=kernel_size, name=name,
                               stride=stride, padding=padding, activation_fn=activation_fn, batch_norm=batch_norm,
                               batch_dropout=batch_dropout, dropout_rate=dropout_rate, kernel_bias_reg=kernel_bias_reg,
                               l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)

            tf.keras.layers.UpSampling1D(size=up_size)(conv_net)
    else:
        with tf.compat.v1.variable_scope('layer_{}'.format(1), values=(x,)):
            #            conv_net = tf.layers.conv1d(x, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)
            conv_net = _d1_cnn(x, filters=filters, kernel_size=kernel_size, name=name,
                               stride=stride, padding=padding, activation_fn=activation_fn, batch_norm=batch_norm,
                               batch_dropout=batch_dropout, dropout_rate=dropout_rate, kernel_bias_reg=kernel_bias_reg,
                               l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)
            tf.keras.layers.UpSampling1D(size=up_size)(conv_net)
            for layer in range(2, n_layers + 1):
                with tf.compat.v1.variable_scope('layer_{}'.format(layer), values=(conv_net,)):
                    #                    conv_net = tf.layers.conv1d(conv_net, filters, kernel_size, strides=stride, use_bias=True, padding='SAME',name=name)
                    conv_net = _d1_cnn(conv_net, filters=filters, kernel_size=kernel_size, name=name,
                                       stride=stride, padding=padding, activation_fn=activation_fn,
                                       batch_norm=batch_norm,
                                       batch_dropout=batch_dropout, dropout_rate=dropout_rate,
                                       kernel_bias_reg=kernel_bias_reg,
                                       l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)
                    tf.keras.layers.UpSampling1D(size=up_size)(conv_net)
    return conv_net


def generator_1d_helper(noise, n_neurons, n_layers,
                        n_cnn_layers, seq_length, feature_dim, filters, kernel_size, pool_size, stride,  # FIXME (davit):  new arguments
                        activation_fn, double_neurons,
                        batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate, is_training):
    # TODO (davit): edit docstring
    """Core ... generator.

    This function is reused between the different GAN anomaly models (...).

    Args:
      noise: A 2D Tensor of shape [batch size, noise dim].

      l2_rate: weight_decay: The value of the l2 weight decay.
      is_training: If `True`, batch norm uses batch statistics. If `False`, batch
        norm uses the exponential moving average collected from population
        statistics.s

    Returns:
      A generated image in the range [-1, 1] if its relu
    """

    g_ff_dense_net = _dense_layers(noise,
                                   n_neurons,
                                   n_layers,
                                   name_scope='G_ff_dense_net',
                                   minibatch_size=0,
                                   activation_fn=activation_fn,
                                   double_neurons=double_neurons,
                                   bottleneck_neurons=False, #TODO (davit): true here doesn't seem reasonalbel as generator needs to upscale noise?
                                   batch_norm=batch_norm,
                                   batch_dropout=batch_dropout,
                                   dropout_rate=dropout_rate,
                                   kernel_bias_reg=kernel_bias_reg,
                                   l1_rate=l1_rate,
                                   l2_rate=l2_rate,
                                   is_training=is_training)

    # g_ff_dense_net = _fullylayer(g_ff_dense_net, seq_length * feature_dim, activation_fn, name='g_ff_dense_net_final',
    #                              batch_norm=batch_norm, batch_dropout=batch_dropout, dropout_rate=dropout_rate,
    #                              kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate, l2_rate=l2_rate,
    #                              is_training=is_training)
    #
    # g_ff_dense_net = tf.reshape(g_ff_dense_net, [-1, seq_length, feature_dim])

    current_dim = g_ff_dense_net.get_shape().as_list()[2]
    target_dim = seq_length * feature_dim

    do_convolute = False
    do_upscale = False
    n_cnn_layers = 1
    if current_dim > target_dim:
        do_convolute = True
        n_cnn_layers = int(current_dim/target_dim)
    else:
        do_upscale = True
        n_cnn_layers = int(target_dim/current_dim)


    g_conv_net = _conv1d(g_ff_dense_net, n_layers=n_cnn_layers, filters=filters, kernel_size=kernel_size, name='g_conv_net',
                         pool_size=pool_size, stride=stride,
                         padding='SAME',
                         activation_fn=None, batch_norm=batch_norm, batch_dropout=batch_dropout,
                         dropout_rate=dropout_rate,
                         kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)

    g_conv_net = _deconv1d(g_conv_net, n_layers, filters, kernel_size, name='g_deconv_net', up_size=4, stride=stride, padding='same',
                  activation_fn=tf.nn.relu, batch_norm=False, batch_dropout=False, dropout_rate=0.0,
                  kernel_bias_reg=False,
                  l1_rate=0.0, l2_rate=0.0, is_training=False)


    # g_ff_dense_net = _dense_layers(g_conv_net,
    #                                n_neurons,
    #                                n_layers,
    #                                name_scope='g_ff_dense_net',
    #                                minibatch_size=0,
    #                                activation_fn=activation_fn,
    #                                double_neurons=False,
    #                                bottleneck_neurons=False,
    #                                batch_norm=batch_norm,
    #                                batch_dropout=batch_dropout,
    #                                dropout_rate=dropout_rate,
    #                                kernel_bias_reg=kernel_bias_reg,
    #                                l1_rate=l1_rate,
    #                                l2_rate=l2_rate,
    #                                is_training=is_training)

    # g_output_logit = tf.compat.v1.layers.dense(G_ff_dense_net, output_dim)
    #
    # g_output_logit = tf.compat.v1.identity(g_output_logit, name="g_output_logit")
    # tf.compat.v1.summary.histogram('g_output_logit', g_output_logit)

    # g_output_logit = tf.layers.dense(G_ff_dense_net, seq_length * feature_dim)
    # g_output_logit = tf.reshape(g_output_logit, [-1, seq_length, feature_dim])  # TODO: seq_length, num_signals

    # TODO (davit): Make sure that generator output is in the same range as `inputs`
    # ??? ie [-1, 1].
    # g_conv_net = tf.tanh(g_conv_net)

    return g_conv_net


# FIXME (davit): work in progress
def discriminator_1d_helper(input, n_neurons, n_layers,
                            output_dim, n_cnn_layers, filters, kernel_size, pool_size, stride, #FIXME (davit):  new arguments
                            activation_fn, double_neurons, bottleneck_neurons,
                            batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate, is_training):
    # TODO (davit): edit docstring
    """Core ... discriminator.

    This function is reused between the different GAN modes (unconditional,
    conditional, etc).

    Args:
      input: Real or generated MNIST digits. Should be in the range [-1, 1].
      l2_rate: weight_decay: The L2 weight decay.

    Returns:
      Final fully connected discriminator layer. [batch_size, ...].
    """

    d_conv_net = _conv1d(input, n_layers=n_cnn_layers, filters=filters, kernel_size=kernel_size, name='d_conv_net',
                        pool_size=pool_size, stride=stride, padding='SAME',
                        activation_fn=None, batch_norm=batch_norm, batch_dropout=batch_dropout,
                        dropout_rate=dropout_rate,
                        kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)

    d_conv_net = tf.reshape(d_conv_net, (-1, d_conv_net.shape[1] * d_conv_net.shape[2]))

    # TODO (davit): what is the activation function here?
    #  leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)

    D_ff_dense_net = _dense_layers(d_conv_net,
                                   n_neurons,
                                   n_layers,
                                   name_scope='D_ff_dense_net',
                                   minibatch_size=1,
                                   activation_fn=activation_fn,
                                   double_neurons=double_neurons,
                                   bottleneck_neurons=bottleneck_neurons,
                                   batch_norm=batch_norm,
                                   batch_dropout=batch_dropout,
                                   dropout_rate=dropout_rate,
                                   kernel_bias_reg=kernel_bias_reg,
                                   l1_rate=l1_rate,
                                   l2_rate=l2_rate,
                                   is_training=is_training)

    d_output_logit = tf.compat.v1.layers.dense(D_ff_dense_net, output_dim,
                                               kernel_regularizer=tf.keras.regularizers.l2(l=l2_rate))

    # TODO (davit): what is the activation function here if we use different ones in hp?
    predictions = tf.nn.sigmoid(d_output_logit)

    return d_output_logit, predictions

def make_encoder_1d(inputs, n_cnn_layers, filters, kernel_size, stride, padding, pool_size, activation_fn, batch_norm, batch_dropout, dropout_rate, kernel_bias_reg, l1_rate, l2_rate, is_training):

        # with tf.variable_scope('Encoder', reuse=reuse):
        #     if denoise is not None:
        #         inputs = tf.nn.dropout(inputs, keep_prob=denoise)
        # output = tf.reshape(inputs, [-1, 1, 64, 64])
        # output = lib.ops.conv2d.Conv2D('Encoder.Input', 1, dim, 3, output, he_init=False)
        #
        # output = ResidualBlock('Encoder.Res1', dim, 2*dim, 3, output, is_training=is_training, resample='down')
        # output = ResidualBlock('Encoder.Res2', 2*dim, 4*dim, 3, output, is_training=is_training, resample='down')
        # output = ResidualBlock('Encoder.Res3', 4*dim, 8*dim, 3, output, is_training=is_training, resample='down')
        # output = ResidualBlock('Encoder.Res4', 8*dim, 8*dim, 3, output, is_training=is_training, resample='down')
        #
        # output = tf.reshape(output, [-1, 4*4*8*dim])
        # output = lib.ops.linear.Linear('Encoder.Output', 4*4*8*dim, z_dim, output)
        # TODO: hyper param on n_layers=1, filters=18, kernel_size=2,  pool_size=2, stride=1,

        encoder = _conv1d(inputs, n_layers=n_cnn_layers, filters=filters, kernel_size=kernel_size, name='encoder_conv_net',
                             pool_size=pool_size, stride=stride,
                             padding=padding,
                             activation_fn=None, batch_norm=batch_norm, batch_dropout=batch_dropout,
                             dropout_rate=dropout_rate,
                             kernel_bias_reg=kernel_bias_reg, l1_rate=l1_rate, l2_rate=l2_rate, is_training=is_training)

        encoder = tf.keras.layers.MaxPooling1D(pool_size, padding=padding)(encoder)

        return encoder
