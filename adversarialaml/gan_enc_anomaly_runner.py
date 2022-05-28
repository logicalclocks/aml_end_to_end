def experiment_wrapper():

    import os
    import sys
    import uuid
    import random
    import numpy as np

    from pydoop import hdfs as pydoop_hdfs
    from hops import hdfs
    from hops import tensorboard
    from hops import model as hops_model

    import tensorflow as tf

    from adversarialaml import gan_enc_anomaly_ff_model
    from adversarialaml import gan_enc_anomaly_ff_trainer
    from adversarialaml import orbit

    ######################################
    NCCL_SOCKET_NTHREADS = '16'
    NCCL_NSOCKS_PERTHREAD = '16'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_NTHREADS'] = NCCL_SOCKET_NTHREADS
    os.environ['NCCL_NSOCKS_PERTHREAD'] = NCCL_NSOCKS_PERTHREAD

    import os
    import sys
    import uuid
    import random

    import tensorflow as tf
    from adversarialaml import keras_utils
    from adversarialaml.gan_enc_ano import GanAnomalyDetector #,  GanAnomalyMonitor
    from hops import tensorboard

    # hops model registry library
    import hsml
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema

    ## Connect to hsfs and retrieve datasets for training and evaluation
    import hsfs
    # Create a connection
    connection = hsfs.connection(engine = "training")
    # Get the feature store handle for the project's feature store
    fs = connection.get_feature_store()

    ben_td = fs.get_training_dataset("gan_non_sar_training_df", 1)
    eval_td = fs.get_training_dataset("gan_eval_df", 1)

    args_dict = {"latent_dim": [8],

                 "discriminator_start_n_units": [1024],
                 "discriminator_n_layers": [128],
                 "discriminator_activation_fn": [2],
                 "discriminator_middle_layer_activation_fn": [2],
                 "discriminator_batch_norm": [0],
                 "discriminator_dropout_rate": [0.001],
                 "discriminator_learning_rate": [0.001],
                 "discriminator_extra_steps": [5],

                 "generator_start_n_units": [1024],
                 "generator_n_layers": [128],
                 "generator_activation_fn": [2],
                 "generator_middle_layer_activation_fn": [2],
                 "generator_batch_norm": [0],
                 "generator_dropout_rate": [0.001],
                 "generator_learning_rate": [0.001],

                 "encoder_start_n_units": [1024],
                 "encoder_n_layers": [128],
                 "encoder_activation_fn": [2],
                 "encoder_middle_layer_activation_fn": [2],
                 "encoder_batch_norm": [0],
                 "encoder_dropout_rate": [0.001],
                 "encoder_learning_rate": [0.001],
                 }

    input_dim=512
    latent_dim = args_dict['latent_dim'][0]

    discriminator_start_n_units = args_dict['discriminator_start_n_units'][0]
    discriminator_n_layers = args_dict['discriminator_n_layers'][0]
    discriminator_activation_fn = args_dict['discriminator_activation_fn'][0]
    discriminator_batch_norm = args_dict['discriminator_batch_norm'][0]
    discriminator_dropout_rate = args_dict['discriminator_dropout_rate'][0]
    discriminator_learning_rate = args_dict['discriminator_learning_rate'][0]
    discriminator_extra_steps = args_dict['discriminator_extra_steps'][0]

    generator_start_n_units = args_dict['generator_start_n_units'][0]
    generator_n_layers = args_dict['generator_n_layers'][0]
    generator_activation_fn = args_dict['generator_activation_fn'][0]
    generator_batch_norm = args_dict['generator_batch_norm'][0]
    generator_dropout_rate = args_dict['generator_dropout_rate'][0]
    generator_learning_rate = args_dict['generator_learning_rate'][0]

    encoder_start_n_units = args_dict['encoder_start_n_units'][0]
    encoder_n_layers = args_dict['encoder_n_layers'][0]
    encoder_activation_fn = args_dict['encoder_activation_fn'][0]
    encoder_batch_norm = args_dict['encoder_batch_norm'][0]
    encoder_dropout_rate = args_dict['encoder_dropout_rate'][0]
    encoder_learning_rate = args_dict['encoder_learning_rate'][0]


    discriminator_middle_layer_activation_fn = args_dict['discriminator_middle_layer_activation_fn'][0]
    generator_middle_layer_activation_fn = args_dict['generator_middle_layer_activation_fn'][0]
    encoder_middle_layer_activation_fn = args_dict['encoder_middle_layer_activation_fn'][0]

    int_to_act_fn = {
        1: 'linear',
        2: 'relu',
        3: 'leaky_relu',
        4: 'selu',
        5: 'tanh'
    }

    discriminator_activation_fn=int_to_act_fn[discriminator_activation_fn]
    discriminator_middle_layer_activation_fn=int_to_act_fn[discriminator_middle_layer_activation_fn]

    if discriminator_dropout_rate > 0.0:
        discriminator_batch_dropout = True
    else:
        discriminator_batch_dropout = False

    if discriminator_dropout_rate > 0.0:
        generator_batch_dropout=True
    else:
        generator_batch_dropout=False

    if encoder_dropout_rate > 0.0:
        encoder_batch_dropout=True
    else:
        encoder_batch_dropout=False

    if discriminator_batch_norm==0:
        discriminator_batch_norm = False
    else:
        discriminator_batch_norm = True

    generator_activation_fn=int_to_act_fn[generator_activation_fn]
    generator_middle_layer_activation_fn=int_to_act_fn[generator_middle_layer_activation_fn]

    if generator_batch_norm==0:
        generator_batch_norm = False
    else:
        generator_batch_norm = True

    encoder_activation_fn=int_to_act_fn[encoder_activation_fn]
    encoder_middle_layer_activation_fn=int_to_act_fn[encoder_middle_layer_activation_fn]

    if encoder_batch_norm==0:
        encoder_batch_norm=False
    else:
        encoder_batch_norm=True

    discriminator_double_neurons=False
    discriminator_bottleneck_neurons=True
    generator_double_neurons=True
    generator_bottleneck_neurons=False

    discriminator_bytes_per_pack= 1 * 1024 * 1024
    generator_bytes_per_pack= 1 * 1024 * 1024
    encoder_bytes_per_pack= 1 * 1024 * 1024

    # Define distribution strategy
    options = tf.distribute.experimental.CommunicationOptions(
        #bytes_per_pack=1 * 1024 * 1024,
        #timeout_seconds=120.0,
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    )
    # Define distribution strategy
    #strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    #strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=options)

    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

    ######################################
    BATCH_SIZE_PER_REPLICA = 8192
    EPOCHS = 10000000

    # Define global batch size
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    TOTAL_SAMPLES = 100000
    STEPS_PER_EPOCH=5000000 #TOTAL_SAMPLES//BATCH_SIZE
    VALIDATION_STEPS=2000

    GP_WEIGHT = 10

    def input_fn(batch_size, epochs, steps_per_epoch):
        x = np.random.random((60000, 512))
        x = tf.cast(x, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(x)
        dataset = dataset.repeat(epochs*steps_per_epoch)
        dataset = dataset.cache()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(50000000) #tf.data.experimental.AUTOTUNE

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        return dataset.with_options(options)

    train_dataset = input_fn(BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH)

    def make_discriminator_model_ff():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(512,)))
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    def make_generator_model_ff():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(8,)))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        return model

    def make_encoder_model_ff():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(512,)))
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        return model

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

    #####################################################
    # construct model under distribution strategy scope
    with strategy.scope():
        discriminator_model = make_discriminator_model_ff()
        generator_model = make_generator_model_ff()
        encoder_model = make_encoder_model_ff()

        """
        discriminator_model = make_discriminator_model_ff(model_name="discriminator_ff", 
                                                                                   input_name="discriminator_ff_inputs", 
                                                                                    input_dim=input_dim, 
                                                                                    output_name="discriminator_ff_output", 
                                                                                    output_dim=1, 
                                                                                    n_units=discriminator_start_n_units,
                                                                                    n_layers=discriminator_n_layers, 
                                                                                    middle_layer_activation_fn=discriminator_middle_layer_activation_fn, 
                                                                                    final_activation_fn_name=discriminator_activation_fn, 
                                                                                    double_neurons=discriminator_double_neurons,
                                                                                    bottleneck_neurons=discriminator_bottleneck_neurons, 
                                                                                    batch_norm=discriminator_batch_norm, 
                                                                                    batch_dropout=discriminator_dropout_rate, 
                                                                                    dropout_rate=discriminator_learning_rate)
        
        generator_model = make_generator_model_ff(model_name = "generator_ff", 
                                                                            input_name = "generator_ff_inputs", 
                                                                            noise_dim = latent_dim, 
                                                                            output_name = "generator_outputs", 
                                                                            output_dim = input_dim, 
                                                                            n_units = generator_start_n_units,
                                                                            n_layers = generator_n_layers, 
                                                                            middle_layer_activation_fn = generator_middle_layer_activation_fn, 
                                                                            final_activation_fn_name = generator_activation_fn, 
                                                                            double_neurons = generator_double_neurons,
                                                                            bottleneck_neurons = generator_bottleneck_neurons, 
                                                                            batch_norm = generator_batch_norm, 
                                                                            batch_dropout = generator_dropout_rate, 
                                                                            dropout_rate = generator_learning_rate)
        
        encoder_model = make_encoder_model_ff(model_name = "encoder_ff", 
                                                                        input_name = "ecoder_ff_inputs", 
                                                                        input_dim = input_dim, 
                                                                        output_name = "generator_outputs", 
                                                                        output_dim = latent_dim, 
                                                                        n_units = encoder_start_n_units,
                                                                        n_layers = encoder_n_layers, 
                                                                        middle_layer_activation_fn = encoder_middle_layer_activation_fn, 
                                                                        final_activation_fn_name = encoder_activation_fn, 
                                                                        double_neurons = False,
                                                                        bottleneck_neurons = True, 
                                                                        batch_norm = encoder_batch_norm, 
                                                                        batch_dropout = encoder_dropout_rate, 
                                                                        dropout_rate = encoder_learning_rate)
        """
        #train_dataset = strategy.distribute_datasets_from_function(
        #  lambda input_context: data_input(train_dataset_files, input_context, BATCH_SIZE, WINDOW_SIZE, EPOCHS))

    # Define optimizers
    with strategy.scope():
        discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        generator_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        encoder_optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    with strategy.scope():
        trainer = gan_enc_anomaly_ff_trainer.GanEncoderAnmalyFfTrainer(
            train_dataset=train_dataset,
            latent_dim=latent_dim,
            input_dim=input_dim,
            d_steps=discriminator_extra_steps,

            global_batch_size=BATCH_SIZE,

            discriminator_model = discriminator_model,
            generator_model = generator_model,
            encoder_model = encoder_model,

            discriminator_loss = discriminator_loss, #gan_enc_anomaly_ff_model.discriminator_loss,
            generator_loss = generator_loss, #gan_enc_anomaly_ff_model.generator_loss,
            encoder_loss = encoder_loss, #gan_enc_anomaly_ff_model.encoder_loss,

            discriminator_optimizer = discriminator_optimizer,
            generator_optimizer = generator_optimizer,
            encoder_optimizer = encoder_optimizer,

            discriminator_bytes_per_pack = discriminator_bytes_per_pack,
            generator_bytes_per_pack = generator_bytes_per_pack,
            encoder_bytes_per_pack = encoder_bytes_per_pack
        )

    controller = orbit.Controller(
        trainer=trainer,
        steps_per_loop=STEPS_PER_EPOCH,
        global_step=trainer.generator_optimizer.iterations)

    from timeit import default_timer as timer
    start = timer()
    controller.train(EPOCHS)
    #end_loss = trainer.train_loss.result().numpy()
    end = timer()
    print(end - start)

    metrics={'val_anomaly_score': 1.0}
    return metrics

from hops import experiment
experiment.mirrored(experiment_wrapper,  name='train_gan_aml', metric_key='val_anomaly_score', local_logdir=False)
