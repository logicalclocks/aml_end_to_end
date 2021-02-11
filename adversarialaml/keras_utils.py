# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Helper functions for the Keras implementations of models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import time

from absl import logging
import tensorflow as tf


class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

  def __repr__(self):
    return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
        self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps
    self.last_log_step = 0
    self.steps_before_epoch = 0
    self.steps_in_epoch = 0
    self.start_time = None
    self.elapsed_time = None

    if logdir:
      self.summary_writer = tf.summary.create_file_writer(logdir)
    else:
      self.summary_writer = None

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

    # Records the time each epoch takes to run from start to finish of epoch.
    self.epoch_runtime_log = []

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  @property
  def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return self.global_steps / sum(self.epoch_runtime_log)

  @property
  def average_examples_per_second(self):
    """The average number of training examples per second across all epochs."""
    return self.average_steps_per_second * self.batch_size

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

    if self.summary_writer:
      self.summary_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
      self.start_time = time.time()

    # Record the timestamp of the first global step
    if not self.timestamp_log:
      self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                               self.start_time))

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
      now = time.time()
      self.elapsed_time = now - self.start_time
      steps_per_second = steps_since_last_log / self.elapsed_time
      examples_per_second = steps_per_second * self.batch_size

      self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
      logging.info(
          'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
          'and %d', self.elapsed_time, examples_per_second, self.last_log_step,
          self.global_steps)

      if self.summary_writer:
        with self.summary_writer.as_default():
          tf.summary.scalar('global_step/sec', steps_per_second,
                            self.global_steps)
          tf.summary.scalar('examples/sec', examples_per_second,
                            self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0


class SimpleCheckpoint(tf.keras.callbacks.Callback):
  """Keras callback to save tf.train.Checkpoints."""

  def __init__(self, checkpoint_manager):
    super(SimpleCheckpoint, self).__init__()
    self.checkpoint_manager = checkpoint_manager

  def on_epoch_end(self, epoch, logs=None):
    step_counter = self.checkpoint_manager._step_counter.numpy()  # pylint: disable=protected-access
    self.checkpoint_manager.save(checkpoint_number=step_counter)


def set_session_config(enable_xla=False):
  """Sets the session config."""
  if enable_xla:
    tf.config.optimizer.set_jit(True)

# TODO(hongkuny): remove set_config_v2 globally.
set_config_v2 = set_session_config


def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                  datasets_num_private_threads,
                                  num_gpus, per_gpu_thread_count):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  logging.info('Logical CPU cores: %s', cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = per_gpu_thread_count or 2
  os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  logging.info('TF_GPU_THREAD_COUNT: %s',
               os.environ['TF_GPU_THREAD_COUNT'])
  logging.info('TF_GPU_THREAD_MODE: %s',
               os.environ['TF_GPU_THREAD_MODE'])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * num_gpus
  num_runtime_threads = num_gpus
  if not datasets_num_private_threads:
    datasets_num_private_threads = min(
        cpu_count - total_gpu_thread_count - num_runtime_threads,
        num_gpus * 8)
    logging.info('Set datasets_num_private_threads to %s',
                 datasets_num_private_threads)



#############
import six
# from tensorflow.python.ops import summary_ops_v2
# from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.profiler import profiler_v2 as profiler


#from  tensorflow.profiler import experimental  as profiler
from tensorflow.python.util import nest

class TimeHistoryWithProfiler(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps, profile_batch=None, logdir=None):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
      logdir: Optional directory to write TensorBoard summaries.
    """
    # TODO(wcromar): remove this parameter and rely on `logs` parameter of
    # on_train_batch_end()
    self.batch_size = batch_size
    super(TimeHistoryWithProfiler, self).__init__()
    self.log_steps = log_steps
    self.last_log_step = 0
    self.steps_before_epoch = 0
    self.steps_in_epoch = 0
    self.start_time = None
    self.elapsed_time = None

    self._train_dir = logdir

    if logdir:
      self.summary_writer = tf.summary.create_file_writer(self._train_dir)
    else:
      self.summary_writer = None

    # Logs start of step 1 then end of each step based on log_steps interval.
    self.timestamp_log = []

    # Records the time each epoch takes to run from start to finish of epoch.
    self.epoch_runtime_log = []

    if profile_batch is not None:
      self.profiler_on = True
      self._init_profile_batch(profile_batch)
    else:
      self.profiler_on = False

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  @property
  def average_steps_per_second(self):
    """The average training steps per second across all epochs."""
    return self.global_steps / sum(self.epoch_runtime_log)

  @property
  def average_epoch_time(self):
    """The average time spent on epochs."""
    return sum(self.epoch_runtime_log) / len(self.epoch_runtime_log)

  @property
  def total_train_time(self):
    """Total training time."""
    return sum(self.epoch_runtime_log)

  @property
  def average_examples_per_second(self):
    """The average number of training examples per second across all epochs."""
    return self.average_steps_per_second * self.batch_size

  def _init_profile_batch(self, profile_batch):
    """Validate profile_batch value and set the range of batches to profile.
    Arguments:
      profile_batch: The range of batches to profile. Should be a non-negative
        integer or a comma separated string of pair of positive integers. A pair
        of positive integers signify a range of batches to profile.
    Returns:
      A pair of non-negative integers specifying the start and stop batch to
      profile.
    Raises:
      ValueError: If profile_batch is not an integer or a comma seperated pair
                  of positive integers.
    """
    profile_batch_error_message = (
      'profile_batch must be a non-negative integer or 2-tuple of positive '
      'integers. A pair of positive integers signifies a range of batches '
      'to profile. Found: {}'.format(profile_batch))

    # Support legacy way of specifying "start,stop" or "start" as str.
    if isinstance(profile_batch, six.string_types):
      profile_batch = str(profile_batch).split(',')
      profile_batch = nest.map_structure(int, profile_batch)

    if isinstance(profile_batch, int):
      self._start_batch = profile_batch
      self._stop_batch = profile_batch
    elif isinstance(profile_batch, (tuple, list)) and len(profile_batch) == 2:
      self._start_batch, self._stop_batch = profile_batch
    else:
      raise ValueError(profile_batch_error_message)

    if self._start_batch < 0 or self._stop_batch < self._start_batch:
      raise ValueError(profile_batch_error_message)

    # if self._start_batch > 0:
    #   profiler.warmup()  # Improve the profiling accuracy.
    # True when a trace is running.
    self._is_tracing = False

    # Setting `profile_batch=0` disables profiling.
    self._should_trace = not (self._start_batch == 0 and self._stop_batch == 0)

  def _start_trace(self):
    #summary_ops_v2.trace_on(graph=True, profiler=False)
    #tf.summary.trace_on(graph=True, profiler=False)
    tf.profiler.experimental.start(logdir=self._train_dir)
    self._is_tracing = True

  def _stop_trace(self, batch=None):
    """Logs the trace graph to TensorBoard."""
    #if batch is None:
    #  batch = self._stop_batch
    # with self.summary_writer.as_default():
    #   # https://github.com/tensorflow/tensorflow/issues/26405
    #   #with summary_ops_v2.always_record_summaries():
    #   #summary_ops_v2.trace_export(name='batch_%d' % batch, step=batch)
    #
    #   tf.summary.trace_export( name='batch_%d' % batch, step=batch)
    tf.profiler.experimental.stop()
    self._is_tracing = False

  # this we will call before train
  def on_train_begin(self, logs=None):
    self.train_start_time = time.time()
    self._global_train_batch = 0
    #self._push_writer(self.summary_writer, self._train_step)


  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

    if self._is_tracing:
      self._stop_trace()

    if self.summary_writer:
      self.summary_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
      self.start_time = time.time()

    self._global_train_batch += 1
    if not self._should_trace:
      return


    if self.profiler_on:
      if self._global_train_batch == self._start_batch:
        self._start_trace()

    # Record the timestamp of the first global step
    if not self.timestamp_log:
      self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                               self.start_time))


  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    if not self._should_trace:
      return

    if self.profiler_on and self._is_tracing and self._global_train_batch >= self._stop_batch:
      self._stop_trace()

    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
      now = time.time()
      self.elapsed_time = now - self.start_time
      steps_per_second = steps_since_last_log / self.elapsed_time
      examples_per_second = steps_per_second * self.batch_size

      self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
      logging.info(
        'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
        'and %d', self.elapsed_time, examples_per_second, self.last_log_step,
        self.global_steps)

      if self.summary_writer:
        with self.summary_writer.as_default():
          tf.summary.scalar('global_step/sec', steps_per_second,
                            self.global_steps)
          tf.summary.scalar('examples/sec', examples_per_second,
                            self.global_steps)

      self.last_log_step = self.global_steps
      self.start_time = None

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    self.epoch_runtime_log.append(epoch_run_time)

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0

