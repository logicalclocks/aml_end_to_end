# Copyright 2020 The Orbit Authors. All Rights Reserved.
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
"""Tests for orbit.standard_runner."""

from adversarialaml.orbit import standard_runner
from adversarialaml.orbit import utils

import tensorflow as tf


def dataset_fn(input_context=None):
  del input_context

  def dummy_data(_):
    return tf.zeros((1, 1), dtype=tf.float32)

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class TestTrainer(standard_runner.StandardTrainer):
  """A StandardTrainer subclass for tests."""

  def __init__(self, options=None):
    self.strategy = tf.distribute.get_strategy()
    self.global_step = utils.create_global_step()
    dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    super().__init__(train_dataset=dataset, options=options)

  def train_loop_begin(self):
    self.global_step.assign(0)

  def train_step(self, iterator):

    def replica_step(_):
      self.global_step.assign_add(1)

    self.strategy.run(replica_step, args=(next(iterator),))

  def train_loop_end(self):
    return self.global_step.numpy()


class TestEvaluator(standard_runner.StandardEvaluator):
  """A StandardEvaluator subclass for tests."""

  def __init__(self, options=None):
    self.strategy = tf.distribute.get_strategy()
    self.global_step = utils.create_global_step()
    dataset = self.strategy.distribute_datasets_from_function(dataset_fn)
    super().__init__(eval_dataset=dataset, options=options)

  def eval_begin(self):
    self.global_step.assign(0)

  def eval_step(self, iterator):

    def replica_step(_):
      self.global_step.assign_add(1)

    self.strategy.run(replica_step, args=(next(iterator),))

  def eval_end(self):
    return self.global_step.numpy()


class StandardRunnerTest(tf.test.TestCase):

  def test_default_trainer(self):
    trainer = TestTrainer()
    self.assertEqual(trainer.train(tf.constant(10)), 10)

  def test_trainer_with_tpu_summary_optimization(self):
    options = standard_runner.StandardTrainerOptions(
        use_tpu_summary_optimization=True)
    trainer = TestTrainer(options)
    self.assertEqual(trainer.train(tf.constant(10)), 10)

  def test_default_evaluator(self):
    evaluator = TestEvaluator()
    self.assertEqual(evaluator.evaluate(tf.constant(10)), 10)


if __name__ == '__main__':
  tf.test.main()
