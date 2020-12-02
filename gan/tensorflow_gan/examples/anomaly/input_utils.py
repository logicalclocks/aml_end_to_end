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

# TODO (davit): revise docstring
"""A utility for evaluating MNIST generative models.

These functions use a pretrained MNIST classifier with ~99% eval accuracy to
measure various aspects of the quality of generated MNIST digits.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import pyod

__all__ = [
    'norm_ano_gen',
    'norm_gen',
    'InputFunction',
    'UpdateGeneratorInputsHook'
]


def norm_ano_gen(input_dim=2, n_samples=100):
    data, target = pyod.utils.data.generate_data(n_train=n_samples, n_test=0, n_features=input_dim, contamination=0.5,
                                                 train_only=True, offset=10, behaviour='old', random_state=None)
    return data.astype(np.float32), target.astype(np.float32)


def norm_gen(input_dim=2, n_samples=100):
    data, target = pyod.utils.data.generate_data(n_train=n_samples, n_test=0, n_features=input_dim, contamination=0.0,
                                                 train_only=True, offset=10, behaviour='old', random_state=None)
    return data.astype(np.float32), target.astype(np.float32)


"""
from tensorflow.python import pywrap_tensorflow

ckpt_path = '/tmp/tfgan_logdir/gan-estimator/model.ckpt-20000'
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()

###############

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = tf.train.latest_checkpoint('/tmp/tfgan_logdir/gan-estimator/')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')

###############

class WarmStartHook(tf.estimator.SessionRunHook):
  def __init__(self, checkpoint):
    self.checkpoint = checkpoint

  def begin(self):
    regex = '.*batch_normalization/(moving.*|gamma|beta):0$'
    #vars_to_warm_start=".*input_layer.*"
    #generator_variables/Generator/G_ff_dense_net
    tf.train.warm_start(self.checkpoint)
    tf.train.warm_start(self.checkpoint, [regex])



class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)

def model_fn():
    ...
    init_fn = assign_from_checkpoint_fn(model_path, var_list, ignore_missing_vars=True)
    ...
    return EstimatorSpec(..., training_hooks=[RestoreHook(init_fn)])
"""


class UpdateGeneratorInputsHook(tf.estimator.SessionRunHook):
    """Hook to update generator inputs after every epoch."""

    def __init__(self, estimator, predict_input, data_size, batch_size, input_dims):
        self._estimator = estimator
        self._pred_input_fn = predict_input
        self.data_size = data_size
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.update_op = None
        self.placeholder = None
        self.g_output_logit = None
        self.g_input = None

    def begin(self):
        self._global_step_tensor = tf.train.get_or_create_global_step()
        self.g_output_logit = tf.get_default_graph().get_tensor_by_name("Generator/g_output_logit:0")
        self.random_normal = tf.random_normal([self.batch_size, self.input_dims], mean=0, stddev=1, name="G_noise_input")

    def before_run(self, run_context):
        requests = {'global_step': self._global_step_tensor, 'g_output_logit': self.g_output_logit, 'random_normal': self.random_normal}
        return tf.estimator.SessionRunArgs(requests)

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""

        self.update_op = session.graph.get_tensor_by_name("g_update:0")
        self.placeholder = session.graph.get_tensor_by_name("g_update_placeholder:0")
        #self.g_output_logit = session.graph.get_tensor_by_name('Generator/g_output_logit_a_1:0')

        #for op in tf.get_default_graph().get_operations():
        #    print(str(op.name))

        #for n in session.graph.as_graph_def().node:
        #    print(n.name)

    def after_run(self, run_context, run_values):

        global_step = run_values.results['global_step']

        # TODO (davit): what is optimal way to recompute generator variables? maybe its better to warmstart generator weights
        self.g_input = run_values.results['random_normal']
        #if self.data_size > global_step:
        #    self.g_input = run_values.results['random_normal']
        #elif int(global_step) % self.data_size == 0:
        #    self.g_input = run_values.results['g_output_logit']

        #predictions = self._estimator.predict(input_fn=self._pred_input_fn, yield_single_examples=False)
        #predictions = np.array([next(predictions) for _ in range(100)])
        #print(predictions.shape)

        #self._estimator.get_variable_value('...')
        feed_dict = {}
        feed_dict[self.placeholder] = self.g_input #predictions[99]
        run_context.session.run(self.update_op, feed_dict=feed_dict)


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        assert callable(self.iterator_initializer_func)
        self.iterator_initializer_func(session)


class InputFunction:
    def __init__(self, data, targets, batch_size, input_dims, num_epochs, mode):
        self.data = data
        self.targets = targets
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.mode = mode
        self.num_epochs = num_epochs
        self.init_hook = IteratorInitializerHook()

    def __call__(self):

        # Define placeholders
        placeholders = [
            tf.compat.v1.placeholder(self.data.dtype, self.data.shape, name='input_tensor'),
            tf.compat.v1.placeholder(self.targets.dtype, self.targets.shape, name='labels_tensor')
        ]

        # Build dataset pipeline
        dataset = tf.data.Dataset.from_tensor_slices(placeholders[0])
        target = tf.data.Dataset.from_tensor_slices(placeholders[1])

        # noise_ds = (tf.data.Dataset.from_tensors(0).repeat().map(lambda _: tf.random.normal([self.input_dims])))
        dataset = tf.data.Dataset.zip((dataset, target))

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat(self.num_epochs)
        dataset = dataset.batch(self.batch_size)

        #------------------------------------------------------------------------------------
        # create iterator from dataset
        next_label = None

        iterator = dataset.make_initializable_iterator()

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            next_example, _ = iterator.get_next()
            next_example = tf.identity(next_example, name="real_input")
        else:
            next_example, next_label = iterator.get_next()
            next_example = tf.identity(next_example, name="real_input")
            next_label = tf.identity(next_label, name="target")

        #g_input = tf.Variable(tf.zeros([self.batch_size, self.input_dims]))
        g_input = tf.compat.v1.get_variable(
            name='g_input',
            initializer=tf.random_normal([self.batch_size, self.input_dims], mean=0, stddev=1, name="G_noise_input"))

        g_update_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.input_dims), name="g_update_placeholder")
        tf.compat.v1.assign(g_input, g_update_placeholder, use_locking=None, name="g_update")
        tf.compat.v1.summary.histogram('g_input', g_input)

        # create initialization hook
        def _init(sess):
            feed_dict = dict(zip(placeholders, [self.data, self.targets]))
            sess.run(iterator.initializer, feed_dict=feed_dict)

        self.init_hook.iterator_initializer_func = _init

        inputs = {'real_input': next_example, 'g_input': g_input.read_value()}

        return inputs, next_label
        #------------------------------------------------------------------------------------
