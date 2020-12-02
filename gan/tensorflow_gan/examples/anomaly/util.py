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
import tensorflow_gan as tfgan

import pyod

__all__ = [
    'anomaly_cross_entropy'
]

def anomaly_cross_entropy(predictions, targets, num_batches=1):
  # TODO (davit): revise docstring
  """Get anomaly cross_entropy with dicriminator classifier score.

  Args:
    images: A minibatch tensor of MNIST digits. Shape must be [batch, 28, 28,
      1].
    num_batches: Number of batches to split `generated_images` in to in order to
      efficiently run them through Inception.

  Returns:
    The classifier score, a floating-point scalar.
  """
  # TODO (davit):
  #images.shape.assert_is_compatible_with([None, 28, 28, 1])

  predictions_reverted = predictions

  val_pred_bool = tf.equal(predictions_reverted, tf.cast(targets, tf.float32))
  val_acc_metric = tf.metrics.mean(val_pred_bool, name="val_acc_metric")
  #val_f1_metric = tf.contrib.metrics.f1_score(tf.cast(targets, tf.float32), tf.cast(predictions_reverted, tf.float32),
  #                                            name="val_f1_metric")

  with tf.compat.v1.name_scope(name='metrics'):
      val_f1_metric = 2 * (tf.compat.v1.metrics.recall(targets, predictions) * tf.compat.v1.metrics.precision(targets,
                                                                                                              predictions)) / (
                                  tf.compat.v1.metrics.recall(targets, predictions) + tf.compat.v1.metrics.precision(
                              targets, predictions))

  val_precision_metric = tf.metrics.precision(tf.cast(targets, tf.float32),
                                              tf.cast(predictions_reverted, tf.float32),
                                              name="val_precision_metric")
  val_recall_metric = tf.metrics.recall(tf.cast(targets, tf.float32),
                                        tf.cast(predictions_reverted, tf.float32), name="val_recall_metric")
  val_specificity_at_sensitivity = tf.metrics.specificity_at_sensitivity(tf.cast(targets, tf.float32),
                                                                         tf.cast(predictions_reverted, tf.float32),
                                                                         sensitivity=0.60,
                                                                         name="val_specificity_at_sensitivity")
  val_sensitivity_at_specificity = tf.metrics.sensitivity_at_specificity(tf.cast(targets, tf.float32),
                                                                         tf.cast(predictions_reverted, tf.float32),
                                                                         specificity=0.60,
                                                                         name="val_sensitivity_at_specificity")
  val_auc = tf.metrics.auc(tf.cast(targets, tf.float32), predictions_reverted)
  val_false_negatives = tf.metrics.false_negatives(tf.cast(targets, tf.float32),
                                                   tf.cast(predictions_reverted, tf.float32))
  val_false_positives = tf.metrics.false_positives(tf.cast(targets, tf.float32),
                                                   tf.cast(predictions_reverted, tf.float32))
  val_recall_at_thresholds = tf.metrics.recall_at_thresholds(tf.cast(targets, tf.float32),
                                                             tf.cast(predictions_reverted, tf.float32),
                                                             thresholds=[0.6])

  #score = tfgan.eval.classifier_score(images, mnist_classifier_fn, num_batches)
  #score.shape.assert_is_compatible_with([])

  return {
            "val_acc_metric": val_acc_metric,
            "val_f1_metric": val_f1_metric,
            "val_precision_metric": val_precision_metric,
            "val_recall_metric": val_recall_metric,
            "val_specificity_at_sensitivity": val_specificity_at_sensitivity,
            "val_sensitivity_at_specificity": val_sensitivity_at_specificity,
            "val_auc": val_auc,
            "val_false_negatives": val_false_negatives,
            "val_false_positives": val_false_positives,
            "val_recall_at_thresholds": val_recall_at_thresholds
  }

################ input function utils ##################

def _parse_function(serialized, input_dim):
      features = {
              'features_array': tf.FixedLenFeature([input_dim], tf.float32),
              'target': tf.FixedLenFeature([], tf.int64)
          }
      # Parse the serialized data so we get a dict with our data.
      parsed_example = tf.parse_single_example(serialized=serialized, features=features)
      # Get the customer_ts as raw bytes.
      customer_ts = parsed_example['features_array']
      label = parsed_example["target"]
      num_classes = 2
      # label = tf.one_hot(parsed_example["label"], num_classes)
      return customer_ts, label

def input_fn(input_files, batch_size, datasize, time_steps, feature_dim, n_epochs, num_parallel_calls):
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(lambda x: _parse_function(x, time_steps, feature_dim), num_parallel_calls)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([time_steps, feature_dim], []),
                                   drop_remainder=True).shuffle(3 * batch_size).repeat(datasize * n_epochs)

    return dataset