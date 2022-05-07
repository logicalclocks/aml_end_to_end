import orbit
import tensorflow as tf

from official.modeling import grad_utils


class GanEncoderAnmalyTrainer(orbit.StandardTrainer):
    """Trains a single-output model on a given dataset.
    This trainer will handle running a model with one output on a single
    dataset. It will apply the provided loss function to the model's output
    to calculate gradients and will apply them via the provided optimizer. It will
    also supply the output of that model to one or more `tf.keras.metrics.Metric`
    objects.
    """

    def __init__(self,
                 train_dataset,
                 discriminator_model,
                 generator_model,
                 encoder_model,
                 discriminator_loss,
                 generator_loss,
                 encoder_loss,
                 discriminator_optimizer,
                 generator_optimizer,
                 encoder_optimizer,
                 global_batch_size,
                 metrics=None,
                 trainer_options=None):
        """Initializes a `SingleTaskTrainer` instance.
        If the `SingleTaskTrainer` should run its model under a distribution
        strategy, it should be created within that strategy's scope.
        This trainer will also calculate metrics during training. The loss metric
        is calculated by default, but other metrics can be passed to the `metrics`
        arg.
        Arguments:
          train_dataset: A `tf.data.Dataset` or `DistributedDataset` that contains a
            string-keyed dict of `Tensor`s.
          model: A `tf.Module` or Keras `Model` object to evaluate. It must accept a
            `training` kwarg.
          loss_fn: A per-element loss function of the form (target, output). The
            output of this loss function will be reduced via `tf.reduce_mean` to
            create the final loss. We recommend using the functions in the
            `tf.keras.losses` package or `tf.keras.losses.Loss` objects with
            `reduction=tf.keras.losses.reduction.NONE`.
          optimizer: A `tf.keras.optimizers.Optimizer` instance.
          metrics: A single `tf.keras.metrics.Metric` object, or a list of
            `tf.keras.metrics.Metric` objects.
          trainer_options: An optional `orbit.utils.StandardTrainerOptions` object.
        """
        self.discriminator_model = discriminator_model
        self.generator_model = generator_model
        self.encoder_model = encoder_model

        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss
        self.encoder_loss = encoder_loss

        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.encoder_optimizer = encoder_optimizer

        ####
        self.latent_dim = [4, 4]
        self.input_dim = [32, 1]
        self.d_steps = 5
        self.gp_weight = 10
        self.beta_cycle_gen=10
        self.global_batch_size=global_batch_size

        # Capture the strategy from the containing scope.
        self.strategy = tf.distribute.get_strategy()

        # We always want to report training loss.
        self.epoch_d_loss_avg = tf.keras.metrics.Mean(name="epoch_d_loss_avg", dtype=tf.float32)
        self.epoch_g_loss_avg = tf.keras.metrics.Mean(name="epoch_g_loss_avg", dtype=tf.float32)
        self.epoch_e_loss_avg = tf.keras.metrics.Mean(name="epoch_e_loss_avg", dtype=tf.float32)
        #self.epoch_a_score_avg = tf.keras.metrics.Mean(name="epoch_a_score_avg", dtype=tf.float32)

        # We need self.metrics to be an iterable later, so we handle that here.
        if metrics is None:
            self.metrics = []
        elif isinstance(metrics, list):
            self.metrics = metrics
        else:
            self.metrics = [metrics]

        super(GanEncoderAnmalyTrainer, self).__init__(
            train_dataset=train_dataset, options=trainer_options)

    def gradient_penalty(self, real_data, fake_data):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated sample
        and added to the discriminator loss.
        """
        # Get the interpolated sample
        real_data_shape = tf.shape(real_data)
        alpha = tf.random.normal(shape=[real_data_shape[0], real_data_shape[1], real_data_shape[2]], mean=0.0, stddev=2.0,
                                 dtype=tf.dtypes.float32)
        # alpha = tf.random_uniform([self.batch_size, 1], minval=-2, maxval=2, dtype=tf.dtypes.float32)
        interpolated = (alpha * real_data) + ((1 - alpha) * fake_data)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated sample.
            pred = self.discriminator_model(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated sample.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[-2, -1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def compute_anomaly_score(self, input):
        # Encode the real data
        encoded_real_data = self.encoder_model(input, training=False)
        # Reconstruct encoded real data
        generator_reconstructed_encoded_real_data = self.generator_model(encoded_real_data, training=False)
        # Calculate distance between real and reconstructed data (Here may be step forward?)
        gen_rec_loss_predict = tf.keras.losses.MeanSquaredError(input, generator_reconstructed_encoded_real_data)

        # # Compute anomaly score
        # real_to_orig_dist_predict = tf.math.reduce_sum(tf.math.pow(encoded_random_latent - encoded_real_data, 2), axis=[-1])
        # anomaly_score = (gen_rec_loss_predict * self.anomaly_alpha) + ((1 - self.anomaly_alpha) * real_to_orig_dist_predict)
        return gen_rec_loss_predict

    def train_loop_begin(self):
        """Actions to take once, at the beginning of each train loop."""
        self.epoch_d_loss_avg.reset_states()
        self.epoch_d_loss_avg.reset_states()
        self.epoch_g_loss_avg.reset_states()
        self.epoch_e_loss_avg.reset_states()
        # TODO (Davit):
        #self.epoch_a_score_avg.reset_states()
        for metric in self.metrics:
            metric.reset_states()

    @tf.function
    def train_step(self, iterator):

        def step_fn(real_data):
            if isinstance(real_data, tuple):
                real_data = real_data[0]

            # Get the batch size
            batch_size = tf.shape(real_data)[0]

            # For each batch, we are going to perform the
            # following steps as laid out in the original paper:
            # 1. Train the generator and get the generator loss
            # 1a. Train the encoder and get the encoder loss
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
                random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1])),
                with tf.GradientTape() as discriminator_tape:
                    # Generate fake data from the latent vector
                    fake_data = self.generator_model(random_latent_vectors, training=True)

                    #(somewhere here step forward?)
                    # Get the logits for the fake data
                    fake_logits = self.discriminator_model(fake_data, training=True)
                    # Get the logits for the real data
                    real_logits = self.discriminator_model(real_data, training=True)

                    # Calculate the discriminator loss using the fake and real sample logits
                    d_cost = self.discriminator_loss(real_sample=real_logits, fake_sample=fake_logits, beta_cycle_gen=self.beta_cycle_gen, global_batch_size=self.global_batch_size)
                    # Calculate the gradient penalty
                    gp = self.gradient_penalty(real_data, fake_data)
                    # Add the gradient penalty to the original discriminator loss
                    d_loss = d_cost + gp * self.gp_weight

                """    
                # Get the gradients w.r.t the discriminator loss
                d_gradient = discriminator_tape.gradient(d_loss, self.discriminator_model.trainable_variables)
                # Update the weights of the discriminator using the discriminator optimizer
                self.discriminator_optimizer.apply_gradients(
                    zip(d_gradient, self.discriminator_model.trainable_variables)
                )
                """
                grad_utils.minimize_using_explicit_allreduce(
                    tape=discriminator_tape,
                    optimizer=self.discriminator_optimizer,
                    loss=d_loss,
                    trainable_variables=self.discriminator_model.trainable_variables,
                    pre_allreduce_callbacks=None,
                    post_allreduce_callbacks=None,
                    allreduce_bytes_per_pack=0)#self.bytes_per_pack

            # Train the generator
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim[0], self.latent_dim[1]))
            with tf.GradientTape() as generator_tape:
                # Generate fake data using the generator
                generated_data = self.generator_model(random_latent_vectors, training=True)
                # Get the discriminator logits for fake data
                gen_sample_logits = self.discriminator_model(generated_data, training=True)
                # Calculate the generator loss
                g_loss = self.generator_loss(gen_sample_logits, self.global_batch_size)

            """    
            # Get the gradients w.r.t the generator loss
            gen_gradient = generator_tape.gradient(g_loss, self.generator_model.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.generator_optimizer.apply_gradients(
                zip(gen_gradient, self.generator_model.trainable_variables)
            )
            """
            grad_utils.minimize_using_explicit_allreduce(
                tape=generator_tape,
                optimizer=self.generator_optimizer,
                loss=g_loss,
                trainable_variables=self.generator_model.trainable_variables,
                pre_allreduce_callbacks=None,
                post_allreduce_callbacks=None,
                allreduce_bytes_per_pack=0)#self.bytes_per_pack

            # Train the encoder
            with tf.GradientTape() as encoder_tape:
                generated_data = self.generator_model(random_latent_vectors, training=True)
                # Compress generate fake data from the latent vector
                encoded_fake_data = self.encoder_model(generated_data, training=True)
                # Reconstruct encoded generate fake data
                generator_reconstructed_encoded_fake_data = self.generator_model(encoded_fake_data, training=True)
                # Encode the latent vector
                encoded_random_latent_vectors = self.encoder_model(tf.random.normal(shape=(batch_size,
                                                                                           self.input_dim[0],
                                                                                           self.input_dim[1])),
                                                                   training=True)
                # Calculate encoder loss
                e_loss = self.encoder_loss(generated_data, generator_reconstructed_encoded_fake_data,
                                           self.global_batch_size)

            """    
            # Get the gradients w.r.t the generator loss
            enc_gradient = encoder_tape.gradient(e_loss, self.encoder_model.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.encoder_optimizer.apply_gradients(
                zip(enc_gradient, self.encoder_model.trainable_variables)
            )
            """
            grad_utils.minimize_using_explicit_allreduce(
                tape=encoder_tape,
                optimizer=self.encoder_optimizer,
                loss=e_loss,
                trainable_variables=self.encoder_model.trainable_variables,
                pre_allreduce_callbacks=None,
                post_allreduce_callbacks=None,
                allreduce_bytes_per_pack=0)#self.bytes_per_pack

            self.epoch_d_loss_avg.update_state(d_loss)
            self.epoch_g_loss_avg.update_state(g_loss)
            self.epoch_e_loss_avg.update_state(e_loss)
            # TODO (Davit)
            #anomaly_score = self.compute_anomaly_score(real_data)
            #self.epoch_a_score_avg.update_state(anomaly_score)

        self.strategy.run(step_fn, args=(next(iterator),))
        #return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train_loop_end(self):
        """Actions to take once after a training loop."""
        with self.strategy.scope():
            # Export the metrics.
            metrics = {metric.name: metric.result() for metric in self.metrics}
            metrics[self.epoch_d_loss_avg.name] = self.epoch_d_loss_avg.result()
            metrics[self.epoch_g_loss_avg.name] = self.epoch_g_loss_avg.result()
            metrics[self.epoch_e_loss_avg.name] = self.epoch_e_loss_avg.result()
            #metrics[self.epoch_a_score_avg.name] = self.epoch_a_score_avg.result()

        return metrics