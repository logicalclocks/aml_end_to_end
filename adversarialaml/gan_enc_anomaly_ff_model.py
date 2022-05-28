#
#   Copyright 2021 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import tensorflow as tf


# Create the discriminator (the critic in the original WGAN)
def make_discriminator_model_ff(model_name, input_name, input_dim, output_name, output_dim, n_units,
                                n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                                bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)


# Create the generator
def make_generator_model_ff(model_name, input_name, noise_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, noise_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)


def make_encoder_model_ff(model_name, input_name, input_dim, output_name, output_dim, n_units,
                          n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                          bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)


def make_decoder_model_ff(model_name, input_name, input_dim, output_name, output_dim, n_units,
                          n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                          bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn_name, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)


def encoder_loss(generated_fake_data, generator_reconstructed_encoded_fake_data, global_batch_size):
    generator_reconstracted_data = tf.cast(generator_reconstructed_encoded_fake_data, tf.float32)
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    per_batch_loss = mse(generated_fake_data, generator_reconstracted_data)
    # per_batch_loss = tf.math.reduce_sum(tf.math.pow(generated_fake_data - generator_reconstracted_data, 2), axis=[-1])
    beta_cycle_gen = 10.0
    per_batch_loss = per_batch_loss * beta_cycle_gen
    # loss = tf.nn.compute_average_loss(per_batch_loss, global_batch_size=BATCH_SIZE)
    return tf.reduce_sum(per_batch_loss) * (1. / global_batch_size)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_sample, fake_sample, beta_cycle_gen, global_batch_size):
    real_loss = tf.reduce_mean(real_sample)
    fake_loss = tf.reduce_mean(fake_sample)
    per_batch_loss = fake_loss - real_loss
    per_batch_loss = per_batch_loss * beta_cycle_gen
    return tf.reduce_sum(per_batch_loss) * (1. / global_batch_size)


# Define the loss functions for the generator.
def generator_loss(fake_sample, global_batch_size):
    per_batch_loss = -tf.reduce_mean(fake_sample)
    return tf.reduce_sum(per_batch_loss) * (1. / global_batch_size)


def _str_to_act_fn(activation_name):
    if activation_name == 'relu':
        activation_fn = tf.keras.layers.Activation(tf.nn.relu)
    elif activation_name == 'leaky_relu':
        activation_fn = tf.keras.layers.Activation(tf.nn.leaky_relu)
    elif activation_name == 'tanh':
        activation_fn = tf.keras.layers.Activation(tf.nn.tanh)
    elif activation_name == 'selu':
        activation_fn = tf.keras.layers.Activation(tf.nn.selu)
    elif activation_name == 'linear':
        activation_fn = None
    elif activation_name is None:
        activation_fn = None
    else:
        raise ("Unknown activation fn " + "`" + activation_name + "`")
    return activation_fn


def _construct_dense_layer(x, units, activation_name, name, batch_norm=False, batch_dropout=False, dropout_rate=0.0):
    activation_fn = _str_to_act_fn(activation_name)

    # Add regularizers when creating variables or layers:
    dense_layer = tf.keras.layers.Dense(
        units=units,
        activation=activation_fn,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    )(x)

    if batch_dropout:
        dense_layer = tf.keras.layers.Dropout(rate=dropout_rate, noise_shape=None, seed=None)(dense_layer)
    if batch_norm:
        dense_layer = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
            fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=name,
        )(dense_layer)

    return dense_layer


def _construct_model(model_name, input_name, input_dim, output_name, output_dim, n_units, n_layers,
                     middle_layer_activation_fn=None, final_activation_fn_name=None, double_neurons=False,
                     bottleneck_neurons=False, batch_norm=True, batch_dropout=False, dropout_rate=0.5):
    inputs = tf.keras.Input(shape=(input_dim,), name=input_name)

    # n_layers must be at least 1
    if n_layers <= 0:
        raise ValueError("n_layers must be at least 1! but at the moment it is" + str(n_layers))
    # n_units must be at least 1
    if n_units <= 0:
        raise ValueError("n_units must be at least 1! but at the moment it is" + str(n_units))

    if double_neurons and bottleneck_neurons:
        raise ValueError("double_neurons and bottleneck_neurons can't both true")

    fc = _construct_dense_layer(inputs, units=n_units, activation_name=middle_layer_activation_fn, name='fc%i' % 1,
                                batch_norm=batch_norm, batch_dropout=batch_dropout, dropout_rate=dropout_rate)

    if n_layers > 1:
        for layer in range(2, n_layers + 1):
            if double_neurons:
                n_units = int(n_units * 2)
            elif bottleneck_neurons:
                n_units = int(n_units / 2)
            fc = _construct_dense_layer(fc, units=n_units, activation_name=middle_layer_activation_fn,
                                        name='fc%i' % layer, batch_norm=batch_norm, batch_dropout=batch_dropout,
                                        dropout_rate=dropout_rate)

    final_activation_fn = _str_to_act_fn(final_activation_fn_name)
    return tf.keras.layers.Dense(output_dim, name=output_name, activation=final_activation_fn)(fc)


def _tmp(
        generator_reconstructed_encoded_fake_data,
        encoded_random_latent_vectors,
        real_data,
        encoded_real_data,
        generator_reconstructed_encoded_real_data,
        alpha=0.7,
        scope="anomaly_score",
        add_summaries=False):
    """anomaly score.
      See https://arxiv.org/pdf/1905.11034.pdf for more details
    """

    with tf.name_scope(scope):
        gen_rec_loss = tf.math.reduce_sum(
            tf.math.pow(real_data - generator_reconstructed_encoded_fake_data, 2), axis=[-2, -1])
        gen_rec_loss_predict = tf.math.reduce_sum(
            tf.math.pow(real_data - generator_reconstructed_encoded_real_data, 2), axis=[-1])
        real_to_orig_dist = tf.math.reduce_sum(
            tf.math.pow(encoded_real_data - encoded_random_latent_vectors, 2), axis=[-2, -1])
        # real_to_orig_dist_predict = tf.math.reduce_sum(
        #    tf.math.pow(encoded_real_data, 2), axis=[-1])

        anomaly_score = (gen_rec_loss_predict * alpha) + ((1 - alpha) * real_to_orig_dist)

        if add_summaries:
            tf.summary.scalar(name=scope + "_gen_rec_loss", data=gen_rec_loss, step=None, description=None)
            tf.summary.scalar(name=scope + "_orig_loss", data=real_to_orig_dist, step=None, description=None)
            tf.summary.scalar(name=scope, data=anomaly_score, step=None, description=None)

    return anomaly_score, gen_rec_loss, real_to_orig_dist, gen_rec_loss_predict,  # real_to_orig_dist_predict


def _dataset_split(ds, target):
    return ds.filter(lambda *x: x[1] == target)


"""
    # define custom server function
    @tf.function
    def serve_function(self, input):
        return self.compute_anomaly_score(input)
"""


class RandomWeightedAverage(tf.keras.layers.Layer):
    """Provides a (random) weighted average between real and generated image samples

        x1 = tf.keras.layers.Input((128, 1))
        x2 = tf.keras.layers.Input((128, 1))

        y = RandomWeightedAverage(4)(inputs=[x1, x2])
        model = tf.keras.Model(inputs=[x1, x2], outputs=[y])
        print(model.summary())
    """

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random_uniform((self.batch_size, 1), minval=-2, maxval=2, dtype=tf.dtypes.float32)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# Create a Keras callback that periodically saves generated data
class GanAnomalyMonitor(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, latent_dim, input_dim, logdir=None):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        if logdir:
            self.summary_writer = tf.summary.create_file_writer(logdir)
        else:
            self.summary_writer = None

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        generated_data = self.model.generator(random_latent_vectors, training=False)
        # Compress generate fake data from the latent vector
        encoded_fake_data = self.model.encoder(generated_data, training=False)
        # Reconstruct encoded generate fake data
        generator_reconstructed_encoded_fake_data = self.model.generator(encoded_fake_data, training=False)
        # Encode the latent vector
        encoded_random_latent_vectors = self.model.encoder(tf.random.normal(shape=(self.batch_size, self.input_dim)),
                                                           training=False)
        if self.summary_writer:
            with self.summary_writer.as_default():
                tf.summary.histogram(name="random_latent_vectors", data=random_latent_vectors, step=epoch,
                                     description=None)
                tf.summary.histogram(name="generated_data", data=generated_data, step=epoch, description=None)
                tf.summary.histogram(name="encoded_fake_data", data=encoded_fake_data, step=epoch, description=None)
                tf.summary.histogram(name="encoded_random_latent_vectors", data=encoded_random_latent_vectors,
                                     step=epoch, description=None)
                tf.summary.histogram(name="generator_reconstructed_encoded_fake_data",
                                     data=generator_reconstructed_encoded_fake_data, step=epoch, description=None)

    def on_train_end(self, logs=None):
        if self.summary_writer:
            self.summary_writer.flush()
