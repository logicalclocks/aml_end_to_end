import os
import multiprocessing

from absl import logging

import math
import orbit

import tensorflow as tf
from official.common import distribute_utils
from official.modeling import performance
from official.utils.misc import keras_utils
from official.vision.image_classification.resnet import common
from official.utils.flags import _performance

from hops import tensorboard
from adversarialaml import resnet_runnable


def set_cudnn_batchnorm_mode(batchnorm_spatial_persistent):
    """Set CuDNN batchnorm mode for better performance.

       Note: Spatial Persistent mode may lead to accuracy losses for certain
       models.
    """
    if batchnorm_spatial_persistent:
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    else:
        os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)

def build_stats(runnable, skip_eval, time_callback):
    """Normalizes and returns dictionary of stats.

    Args:
      runnable: The module containing all the training and evaluation metrics.
      time_callback: Time tracking callback instance.

    Returns:
      Dictionary of normalized results.
    """
    stats = {}

    if not skip_eval:
        stats['eval_loss'] = runnable.test_loss.result().numpy()
        stats['eval_acc'] = runnable.test_accuracy.result().numpy()

        stats['train_loss'] = runnable.train_loss.result().numpy()
        stats['train_acc'] = runnable.train_accuracy.result().numpy()

    if time_callback:
        #timestamp_log = time_callback.timestamp_log
        #stats['step_timestamp_log'] = timestamp_log
        stats['train_start_time'] = time_callback.train_start_time
        stats['train_finish_time'] = time_callback.train_finish_time
        stats['total_train_time'] = time_callback.total_train_time
        stats['average_epoch_time'] = time_callback.average_epoch_time
        if time_callback.epoch_runtime_log:
            stats['avg_exp_per_second'] = time_callback.average_examples_per_second

    return stats

def get_datasets_num_private_threads(datasets_num_private_threads, num_gpus, per_gpu_thread_count):
    """Set GPU thread mode and count, and adjust dataset threads count."""
    cpu_count = multiprocessing.cpu_count()
    logging.info('Logical CPU cores: %s', cpu_count)

    per_gpu_thread_count = per_gpu_thread_count or 2
    #private_threads = (cpu_count -  strategy.num_replicas_in_sync * (per_gpu_thread_count + per_gpu_thread_count))
    num_runtime_threads = num_gpus

    total_gpu_thread_count = per_gpu_thread_count * num_gpus
    if not datasets_num_private_threads:
        datasets_num_private_threads = min(
            cpu_count - total_gpu_thread_count - num_runtime_threads,
            num_gpus * 8)
        logging.info('Set datasets_num_private_threads to %s',
                     datasets_num_private_threads)
    return datasets_num_private_threads

def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                      datasets_num_private_threads,
                                      num_gpus, per_gpu_thread_count):

    # Allocate private thread pool for each GPU to schedule and launch kernels
    per_gpu_thread_count = per_gpu_thread_count or 2
    os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
    os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    logging.info('TF_GPU_THREAD_COUNT: %s',
                 os.environ['TF_GPU_THREAD_COUNT'])
    logging.info('TF_GPU_THREAD_MODE: %s',
                 os.environ['TF_GPU_THREAD_MODE'])

def get_num_train_iterations_non_flag(batch_size, train_epochs, num_images_train, num_images_val, provided_train_steps=None):

    """Returns the number of training steps, train and test epochs."""
    train_steps = (
            num_images_train // batch_size)
    train_epochs = train_epochs

    if provided_train_steps:
        train_steps = min(provided_train_steps, train_steps)
        train_epochs = 1

    eval_steps = math.ceil(1.0 * num_images_val / batch_size)

    return train_steps, train_epochs, eval_steps


def run(
        num_gpus = 16,

        global_batch_size = 2048,
        train_epochs = 300,
        epochs_between_evals = 1,
        steps_per_loop = 100,
        log_steps = 2,

        enable_xla = False, # this crashes collective ops? TODO: need to debug why, might be bug in nightly
        single_l2_loss_op = True, #False
        #         fp16_implementation = 'graph_rewrite',
        fp16_implementation = 'keras',
        dtype= 'fp16',
        loss_scale = None, #'dynamic',
        num_packs = 4, # this is releveant only for 'mirrored': _mirrored_cross_device_ops
        bytes_per_pack = 32 * 1024 * 1024, # this is releveant only for 'multi_worker_mirrored': for collective hints

        tf_gpu_thread_mode = 'gpu_private', #None,
        batchnorm_spatial_persistent = True,
        #distribution_strategy = 'mirrored',
        distribution_strategy = 'multi_worker_mirrored',
        #distribution_strategy = 'one_device',
        all_reduce_alg = 'nccl',
        NCCL_SOCKET_NTHREADS = '16',
        NCCL_NSOCKS_PERTHREAD = '16',
        use_synthetic_data = True,
        data_dir = None,
        data_format = None,
        use_tf_while_loop = True,
        enable_checkpoint_and_export = False,
        enable_tensorboard = True,
        skip_eval = False,
        use_tf_function = True,
        per_gpu_thread_count = 2,
        num_images_train = 100000,
        num_images_val = 20000,
        datasets_num_private_threads = None
):
    """Run ResNet ImageNet training and eval loop using custom training loops.
    Args:
      flags_obj: An object containing parsed flag values.
    Raises:
      ValueError: If fp16 is passed as it is not currently supported.
    Returns:
      Dictionary of training and eval stats.
    """

    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_NTHREADS'] = NCCL_SOCKET_NTHREADS
    os.environ['NCCL_NSOCKS_PERTHREAD'] = NCCL_NSOCKS_PERTHREAD

    dtype = _performance.DTYPE_MAP[dtype]
    datasets_num_private_threads = get_datasets_num_private_threads(datasets_num_private_threads, num_gpus, per_gpu_thread_count)
    keras_utils.set_session_config()
    performance.set_mixed_precision_policy(dtype)

    model_dir = tensorboard.logdir()

    if tf.config.list_physical_devices('GPU'):
        if tf_gpu_thread_mode:
            keras_utils.set_gpu_thread_mode_and_count(
                per_gpu_thread_count=per_gpu_thread_count,
                gpu_thread_mode=tf_gpu_thread_mode,
                num_gpus=num_gpus,
                datasets_num_private_threads=datasets_num_private_threads)
        common.set_cudnn_batchnorm_mode()

    if data_format is None:
        data_format = ('channels_first' if tf.config.list_physical_devices('GPU')
                       else 'channels_last')
    tf.keras.backend.set_image_data_format(data_format)

    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=distribution_strategy,
        num_gpus=num_gpus,
        all_reduce_alg=all_reduce_alg,
        num_packs=num_packs,
        tpu_address=None)

    per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations_non_flag(global_batch_size, train_epochs,
                                                                                  num_images_train,
                                                                                  num_images_val,
                                                                                  provided_train_steps=None)
    if steps_per_loop is None:
        steps_per_loop = per_epoch_steps
    elif steps_per_loop > per_epoch_steps:
        steps_per_loop = per_epoch_steps
        logging.warn('Setting steps_per_loop to %d to respect epoch boundary.',
                     steps_per_loop)
    else:
        steps_per_loop = steps_per_loop

    logging.info(
        'Training %d epochs, each epoch has %d steps, '
        'total steps: %d; Eval %d steps', train_epochs, per_epoch_steps,
        train_epochs * per_epoch_steps, eval_steps)

    time_callback = keras_utils.TimeHistory(
        global_batch_size,
        log_steps,
        logdir=model_dir if enable_tensorboard else None)

    if tf_gpu_thread_mode:
        set_gpu_thread_mode_and_count(
            per_gpu_thread_count=per_gpu_thread_count,
            gpu_thread_mode=tf_gpu_thread_mode,
            num_gpus=num_gpus,
            datasets_num_private_threads=datasets_num_private_threads)
        set_cudnn_batchnorm_mode(batchnorm_spatial_persistent)

    with distribute_utils.get_strategy_scope(strategy):
        runnable = resnet_runnable.ResnetRunnable(
            use_synthetic_data,
            data_dir,
            use_tf_while_loop,
            use_tf_function,
            dtype,
            global_batch_size,
            datasets_num_private_threads,
            single_l2_loss_op,
            loss_scale,
            fp16_implementation,
            bytes_per_pack,
            num_images_train,
            time_callback,
            per_epoch_steps,
            enable_xla,
            skip_eval
        )

    eval_interval = epochs_between_evals * per_epoch_steps
    checkpoint_interval = (
        steps_per_loop * 5 if enable_checkpoint_and_export else None)
    summary_interval = steps_per_loop if enable_tensorboard else None

    checkpoint_manager = tf.train.CheckpointManager(
        runnable.checkpoint,
        directory=model_dir,
        max_to_keep=10,
        step_counter=runnable.global_step,
        checkpoint_interval=checkpoint_interval)

    resnet_controller = orbit.Controller(
        strategy=strategy,
        trainer=runnable,
        evaluator=runnable if not skip_eval else None,
        global_step=runnable.global_step,
        steps_per_loop=steps_per_loop,
        checkpoint_manager=checkpoint_manager,
        summary_interval=summary_interval,
        summary_dir=model_dir,
        eval_summary_dir=model_dir)

    time_callback.on_train_begin()
    if not skip_eval:
        resnet_controller.train_and_evaluate(
            train_steps=per_epoch_steps * train_epochs,
            eval_steps=eval_steps,
            eval_interval=eval_interval)
    else:
        resnet_controller.train(steps=per_epoch_steps * train_epochs)
    time_callback.on_train_end()

    stats = build_stats(runnable, skip_eval, time_callback)
    return stats
