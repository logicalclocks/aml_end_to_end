# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors and Swedbank AI team.
#
# This project is fork of tfgan and extended by Swedbank AI team.
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

# python2 python3
"""Loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

__all__ = [
    'provide_data_from_tfrecord_files'
]

def _to_int32(tensor):
    return tf.cast(tensor, tf.int32)



def provide_data_from_tfrecord_files(input_files, time_steps, feature_dim, batch_size, padded_batch, timeseries=False):

    """

    :param dir_name:
    :param file_name:
    :param file_system:
    :param file_system:
    :param time_steps: hops or local files system
    :param feature_dim:
    :param batch_size: The number of images in each minibatch.  Defaults to 32.
    :param num_parallel_calls:
    :param shuffle: Whether to shuffle.

    Returns:
    A tf.data.Dataset with:
      * data: A `Tensor` of size [batch_size, ..., ..., ...] and type tf.float32.

    """

    def _standard_ds_pipeline():

        """Efficiently process and batch a tf.data.Dataset."""

        def _tfrecord_parser(serialized, time_steps, feature_dim):
            """
            :param serialized:
            :param time_steps:
            :param feature_dim:
            :param timeseries:
            :return:
            """
            array_length = time_steps * feature_dim
            features = \
                {
                    'features': tf.io.FixedLenFeature([array_length], tf.float32),
                    'target': tf.io.FixedLenFeature([], tf.float32)
                }
            # Parse the serialized data so we get a dict with our data.
            parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
            # Get the customer_ts as raw bytes.
            customer_ts = parsed_example['features']
            label = parsed_example["target"]
            return customer_ts, label

        ds = tf.data.Dataset.list_files(input_files)
        ds = ds.interleave(tf.compat.v1.data.TFRecordDataset, cycle_length=4, block_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda x: _tfrecord_parser(x, time_steps, feature_dim), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .cache() \
            .repeat()

        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        # FIXME (davit): this is not working
        if padded_batch:
            if timeseries:
                ds = ds.padded_batch(batch_size, padded_shapes=([time_steps, feature_dim], []), drop_remainder=True)
            else:
                ds = ds.padded_batch(batch_size, padded_shapes=([time_steps * feature_dim], []), drop_remainder=True)
        else:
            ds.batch(batch_size, drop_remainder=True)

        return ds

    return _standard_ds_pipeline



class IteratorInitializerHook(tf.compat.v1.estimator.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


# TODO (davit): remove this is not used any more
class InputFunction:
    def __init__(self, dataset, features_type, targets_type, features_shape, targets_shape):
        self.dataset = dataset
        self.features_type = features_type
        self.targets_type = targets_type
        self.features_shape = features_shape
        self.targets_shape = targets_shape

        self.init_hook = IteratorInitializerHook()

    def __call__(self):

        # Define placeholders
        placeholders = [
            tf.compat.v1.placeholder(self.features_type, self.features_shape, name='input_tensor'),
            tf.compat.v1.placeholder(self.features_type, self.targets_shape, name='labels_tensor')
        ]

        # create iterator from dataset
        iterator = self.dataset.make_initializable_iterator()

        next_label = None
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            next_example, _ = iterator.get_next()
            next_example = tf.identity(next_example, name="real_input")
        else:
            next_example, next_label = iterator.get_next()
            next_example = tf.identity(next_example, name="real_input")
            next_label = tf.identity(next_label, name="target")

        # g_input = tf.Variable(tf.zeros([self.batch_size, self.input_dims]))
        g_input = tf.compat.v1.get_variable(
            name='g_input',
            initializer=tf.random.normal([self.batch_size, self.input_dims], mean=0, stddev=1, name="G_noise_input"))

        g_update_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.input_dims),
                                                        name="g_update_placeholder")
        tf.compat.v1.assign(g_input, g_update_placeholder, use_locking=None, name="g_update")
        tf.compat.v1.summary.histogram('g_input', g_input)

        # create initialization hook
        def _init(sess):
            #            feed_dict = dict(zip(placeholders, [self.data, self.targets]))
            feed_dict = dict(zip(placeholders, self.dataset))
            sess.run(iterator.initializer, feed_dict=feed_dict)

        self.init_hook.iterator_initializer_func = _init

        inputs = {'real_input': next_example, 'g_input': g_input.read_value()}

        return (inputs, next_label)

class UpdateGeneratorInputsHook(tf.compat.v1.estimator.SessionRunHook):
    """Hook to update generator inputs after every epoch."""

    def __init__(self, estimator, predict_input, data_size, batch_size, input_dims, experiment_type, feature_dim=None, seq_length=None,
                 d1_net=False):
        self._estimator = estimator
        self._pred_input_fn = predict_input
        self.data_size = data_size
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.experiment_type = experiment_type
        self.update_op = None
        self.placeholder = None
        self.g_output_logit = None
        self.g_input = None
        # extra variables for 1d cnn gan
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.d1_net = d1_net

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_or_create_global_step()
        self.g_output_logit = tf.compat.v1.get_default_graph().get_tensor_by_name("Generator/g_output_logit:0")
        if self.d1_net:
            self.random_normal = tf.random.normal([self.batch_size, self.seq_length, self.feature_dim], mean=0, stddev=1,
                                        dtype=tf.float32, name="G_noise_input")
        else:
            self.random_normal = tf.random.normal([self.batch_size, self.input_dims], mean=0, stddev=1,
                                                  name="G_noise_input")

    def before_run(self, run_context):
        requests = {'global_step': self._global_step_tensor, 'g_output_logit': self.g_output_logit,
                    'random_normal': self.random_normal}
        return tf.compat.v1.estimator.SessionRunArgs(requests)

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""

        #----------
        # FIXME (davit):
        # (simone) In order to make it compatible with distributed training, the except was added
        # Update_0 is the default name_scope that is assigned to g_update operation in case of distributed training.

        if self.experiment_type == 'hp':
          self.update_op = session.graph.get_tensor_by_name("g_update:0")
        elif self.experiment_type == 'train':
            self.update_op = session.graph.get_operation_by_name("update_0/g_update")
        
        self.placeholder = session.graph.get_tensor_by_name("g_update_placeholder:0")

    def after_run(self, run_context, run_values):
        global_step = run_values.results['global_step']

        # TODO (davit): what is optimal way to recompute generator variables? maybe its better to warmstart generator weights
        self.g_input = run_values.results['random_normal']

        feed_dict = {}
        feed_dict[self.placeholder] = self.g_input  # predictions[99]

        run_context.session.run(self.update_op, feed_dict=feed_dict)