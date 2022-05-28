import tensorflow as tf


def make_discriminator_model_cnn(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim[0], input_dim[1]))
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=1, padding='same', kernel_initializer="uniform")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='same', kernel_initializer="uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # dense output layer
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    prediction = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=prediction)


def make_generator_model_cnn(input_dim, latent_dim):
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim[0], latent_dim[1]))
    x = tf.keras.layers.Conv1D(filters=4, kernel_size=1, padding='same', kernel_initializer="uniform")(latent_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=1, padding='same', kernel_initializer="uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same', kernel_initializer="uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1D(filters=input_dim[1], kernel_size=1, padding='same', kernel_initializer="uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return tf.keras.Model(inputs=latent_inputs, outputs=x)


def make_encoder_model_cnn(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim[0], input_dim[1]))
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding='same', kernel_initializer="uniform")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=1, padding='same', kernel_initializer="uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.Conv1D(filters=4, kernel_size=1, padding='same', kernel_initializer="uniform")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


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
