import tensorflow as tf


# Create the GanAnomalyDetector model
class GanAnomalyDetector(tf.keras.Model):
    def __init__(
            self,
            input_dim,
            latent_dim,

            discriminator_start_n_units,
            discriminator_n_layers,
            discriminator_activation_fn,
            discriminator_double_neurons,
            discriminator_bottleneck_neurons,
            discriminator_batch_norm,
            discriminator_batch_dropout,
            discriminator_dropout_rate,
            discriminator_learning_rate,
            discriminator_extra_steps,  # 3

            generator_start_n_units,
            generator_n_layers,
            generator_activation_fn,
            generator_double_neurons,
            generator_bottleneck_neurons,
            generator_batch_norm,
            generator_batch_dropout,
            generator_dropout_rate,
            generator_learning_rate,

            encoder_start_n_units,
            encoder_n_layers,
            encoder_activation_fn,
            encoder_batch_norm,
            encoder_batch_dropout,
            encoder_dropout_rate,
            encoder_learning_rate,

            gp_weight=10.0,
    ):
        super(GanAnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.encoder_learning_rate = encoder_learning_rate


        self.discriminator = get_discriminator_model(model_name="discriminator", input_name="real_inputs",
                                                     input_dim=self.input_dim, output_name="discriminator_outputs",
                                                     output_dim=1, n_units=discriminator_start_n_units,
                                                     n_layers=discriminator_n_layers, middle_layer_activation_fn=None,
                                                     final_activation_fn=discriminator_activation_fn,
                                                     double_neurons=discriminator_double_neurons,
                                                     bottleneck_neurons=discriminator_bottleneck_neurons,
                                                     batch_norm=discriminator_batch_norm,
                                                     batch_dropout=discriminator_batch_dropout,
                                                     dropout_rate=discriminator_dropout_rate)

        self.generator = get_generator_model(model_name="generator", input_name="fake_inputs",
                                             noise_dim=self.latent_dim, output_name="generator_outputs",
                                             output_dim=self.input_dim, n_units=generator_start_n_units,
                                             n_layers=generator_n_layers, middle_layer_activation_fn=None,
                                             final_activation_fn=generator_activation_fn,
                                             double_neurons=generator_double_neurons,
                                             bottleneck_neurons=generator_bottleneck_neurons,
                                             batch_norm=generator_batch_norm, batch_dropout=generator_batch_dropout,
                                             dropout_rate=generator_dropout_rate)

        self.encoder = get_encoder_model(model_name="encoder", input_name="encoder_inputs", input_dim=self.input_dim,
                                         output_name="encoder_outputs", output_dim=self.latent_dim,
                                         n_units=encoder_start_n_units, n_layers=encoder_n_layers,
                                         middle_layer_activation_fn=None, final_activation_fn=encoder_activation_fn,
                                         double_neurons=False, bottleneck_neurons=True, batch_norm=encoder_batch_norm,
                                         batch_dropout=encoder_batch_dropout, dropout_rate=encoder_dropout_rate)

    def compile(self):
        super(GanAnomalyDetector, self).compile()
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.discriminator_learning_rate, beta_1=0.5,
                                                    beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_learning_rate, beta_1=0.5, beta_2=0.9)
        self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=self.encoder_learning_rate, beta_1=0.5, beta_2=0.9)
        self.d_loss_fn = _discriminator_loss
        self.g_loss_fn = _generator_loss
        self.e_loss_fn = _encoder_loss

    def model_summaries(self, model):
        if model == "discriminator":
            self.discriminator.summay()
        elif model == "generator":
            self.generator.summay()
        elif model == "encoder":
            self.encoder.summay()
        else:
            raise ValueError("Unknown model name, must be 'discriminator', 'generator' or 'encoder'")

    def gradient_penalty(self, real_data, fake_data):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated sample
        and added to the discriminator loss.
        """
        # Get the interpolated sample
        diff = fake_data - real_data
        batch_size = (tf.compat.dimension_value(diff.shape.dims[0]) or
                      tf.shape(input=diff)[0])
        alpha_shape = [batch_size] + [1] * (diff.shape.ndims - 1)
        alpha = tf.random.normal(shape=alpha_shape, mean=0.0, stddev=1.0, )
        interpolated = real_data + (alpha * diff)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated sample.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated sample.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[-2, -1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_data):
        if isinstance(real_data, tuple):
            real_data = real_data[0]

        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake data from the latent vector
                fake_data = self.generator(random_latent_vectors, training=True)

                # Get the logits for the fake data
                fake_logits = self.discriminator(fake_data, training=True)
                # Get the logits for the real data
                real_logits = self.discriminator(real_data, training=True)

                # Calculate the discriminator loss using the fake and real sample logits
                d_cost = self.d_loss_fn(real_sample=real_logits, fake_sample=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(real_data, fake_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake data using the generator
            generated_data = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake data
            gen_sample_logits = self.discriminator(generated_data, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_sample_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Train the encoder
        with tf.GradientTape() as tape:
            generated_data = self.generator(random_latent_vectors, training=True)
            # Compress generate fake data from the latent vector
            encoded_fake_data = self.encoder(generated_data, training=True)
            # Reconstruct encoded generate fake data
            generator_reconstructed_encoded_fake_data = self.generator(encoded_fake_data, training=True)
            # Encode the latent vector
            encoded_random_latent_vectors = self.encoder(tf.random.normal(shape=(batch_size, self.input_dim)),
                                                         training=True)
            # Calculate encoder loss
            e_loss = self.e_loss_fn(generated_data, generator_reconstructed_encoded_fake_data)

        # Get the gradients w.r.t the generator loss
        enc_gradient = tape.gradient(e_loss, self.encoder.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.e_optimizer.apply_gradients(
            zip(enc_gradient, self.encoder.trainable_variables)
        )

        return {"d_loss": d_loss, "g_loss": g_loss, "e_loss": e_loss}


# Create a Keras callback that periodically saves generated data
class GanAnomalyMonitor(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, latent_dim, input_dim, alpha, real_data):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.real_data = real_data
        self.alpha = alpha

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_sample, self.latent_dim))
        generated_data = self.model.generator(random_latent_vectors, training=False)
        # Compress generate fake data from the latent vector
        encoded_fake_data = self.model.encoder(generated_data, training=False)
        # Reconstruct encoded generate fake data
        generator_reconstructed_encoded_fake_data = self.model.generator(encoded_fake_data, training=False)
        # Encode the latent vector
        encoded_random_latent_vectors = self.encoder(tf.random.normal(shape=(self.batch_size, self.input_dim)),
                                                     training=False)
        # Encode the latent vector
        encoded_real_data = self.model.encoder(self.real_data, training=False)
        # Reconstruct encoded generate fake data
        generator_reconstructed_encoded_real_data = self.model.generator(encoded_real_data, training=False)

        _compute_anomaly_score(
            generator_reconstructed_encoded_fake_data,
            encoded_random_latent_vectors,
            self.real_data,
            encoded_real_data,
            generator_reconstructed_encoded_real_data,
            self.alpha,
            scope="anomaly_score",
            add_summaries=True)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def _discriminator_loss(real_sample, fake_sample):
    real_loss = tf.reduce_mean(real_sample)
    fake_loss = tf.reduce_mean(fake_sample)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def _generator_loss(fake_sample):
    return -tf.reduce_mean(fake_sample)


def _encoder_loss(
        generated_fake_data,
        generator_reconstructed_encoded_fake_data):
    generator_reconstracted_data = tf.cast(generator_reconstructed_encoded_fake_data, tf.float32)
    # loss = tf.reduce_sum(tf.pow(generated_fake_data - generator_reconstracted_data, 2), axis=[-2, -1])
    # beta_cycle_gen = 10.0
    # loss = loss * beta_cycle_gen
    loss = tf.reduce_mean(tf.math.abs(generated_fake_data - generator_reconstracted_data))
    return loss


def _construct_dense_layer(x, units, activation_fn, name, batch_norm=False, batch_dropout=False, dropout_rate=0.0):
    if activation_fn == 'relu':
        activation_fn = tf.keras.layers.Activation(tf.nn.relu)
    elif activation_fn == 'leaky_relu':
        activation_fn = tf.keras.layers.Activation(tf.nn.leaky_relu)
    elif activation_fn == 'tanh':
        activation_fn = tf.keras.layers.Activation(tf.nn.tanh)
    elif activation_fn == 'selu':
        activation_fn = tf.keras.layers.Activation(tf.nn.selu)
    elif activation_fn == 'linear':
        activation_fn = None

    # Add regularizers when creating variables or layers:
    dense_layer = tf.keras.layers.Dense(
        units=units, activation=activation_fn, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
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
                     middle_layer_activation_fn=None, final_activation_fn=None, double_neurons=False,
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

    fc = _construct_dense_layer(inputs, units=n_units, activation_fn=middle_layer_activation_fn, name='fc%i' % 1,
                                batch_norm=batch_norm, batch_dropout=batch_dropout, dropout_rate=dropout_rate)

    if n_layers > 1:
        for layer in range(2, n_layers + 1):
            if double_neurons:
                n_units = int(n_units * 2)
            elif bottleneck_neurons:
                n_units = int(n_units / 2)
            fc = _construct_dense_layer(fc, units=n_units, activation_fn=middle_layer_activation_fn,
                                        name='fc%i' % layer, batch_norm=batch_norm, batch_dropout=batch_dropout,
                                        dropout_rate=dropout_rate)

    outputs = tf.keras.layers.Dense(output_dim, name=output_name, activation=final_activation_fn)(fc)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def _compute_anomaly_score(
        generator_reconstructed_encoded_fake_data,
        encoded_random_latent_vectors,
        real_data,
        encoded_real_data,
        generator_reconstructed_encoded_real_data,
        alpha,
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
        real_to_orig_dist_predict = tf.math.reduce_sum(
            tf.math.pow(encoded_real_data, 2), axis=[-1])

        anomaly_score = (gen_rec_loss_predict * alpha) + ((1 - alpha) * real_to_orig_dist_predict)

        if add_summaries:
            tf.summary.scalar(name=scope + "_gen_rec_loss", data=gen_rec_loss, step=None, description=None)
            tf.summary.scalar(name=scope + "_orig_loss", data=real_to_orig_dist, step=None, description=None)
            tf.summary.scalar(name=scope, data=anomaly_score, step=None, description=None)

    return anomaly_score, gen_rec_loss, real_to_orig_dist, gen_rec_loss_predict, real_to_orig_dist_predict


# Create the discriminator (the critic in the original WGAN)
def get_discriminator_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)


# Create the generator
def get_generator_model(model_name, input_name, noise_dim, output_name, output_dim, n_units,
                        n_layers, middle_layer_activation_fn, final_activation_fn, double_neurons,
                        bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, noise_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)


def get_encoder_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                      n_layers, middle_layer_activation_fn, final_activation_fn, double_neurons,
                      bottleneck_neurons, batch_norm, batch_dropout, dropout_rate):
    return _construct_model(model_name, input_name, input_dim, output_name, output_dim, n_units,
                            n_layers, middle_layer_activation_fn, final_activation_fn, double_neurons,
                            bottleneck_neurons, batch_norm, batch_dropout, dropout_rate)
