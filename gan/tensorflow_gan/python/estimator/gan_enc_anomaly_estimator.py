# coding=utf-8
# Copyright 2019 The Swedbank GAN Anomaly detection Authors.
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

"""A TF-GAN-backed GAN Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect
import enum

import tensorflow as tf

from tensorflow_gan.python import namedtuples as tfgan_tuples
from tensorflow_gan.python import train as tfgan_train
from tensorflow_gan.python.eval import summaries as tfgan_summaries

__all__ = [
    'GANEncAnomalyEstimator',
    'SummaryType'
]

Optimizers = collections.namedtuple('Optimizers', ['gopt', 'dopt', 'eopt'])


class SummaryType(enum.IntEnum):
    NONE = 0
    VARIABLES = 1


summary_type_map = {
    SummaryType.VARIABLES:
        tfgan_summaries.add_gan_anomaly_model_summaries
}


class GANEncAnomalyEstimator(tf.estimator.Estimator):
    # TODO (davit): revise docstring
    """An estimator for Anomaly detection using Generative Adversarial Networks (GANs).

  This Estimator is backed by TF-GAN. The network functions follow the TF-GAN
  API except for one exception: if either `generator_fn` or `discriminator_fn`
  have an argument called `mode`, then the tf.Estimator mode is passed in for
  that argument. This helps with operations like batch normalization, which have
  different train and evaluation behavior.

  Example:

  ```python
      import tensorflow as tf
      import tensorflow_gan as tfgan

      # See TF-GAN's `train.py` for a description of the generator and
      # discriminator API.

      def generator_fn(generator_inputs):
        ...
        return generated_data

      def encoder_fn(generated_data):
        ...
        return encoded_generated_data

      def encoder_fn(data):
        ...
        return encoded_data

      def discriminator_fn(data):
        ...
        return logits

      def discriminator_fn(generated_data):
        ...
        return logits


      # Create GAN estimator.
      gan_estimator = tfgan.estimator.GANEncAnomalyEstimator(
          model_dir,
          generator_fn=generator_fn,
          discriminator_fn=discriminator_fn,
          encoder_fn=encoder_fn,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
          encoder_loss=...
          generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
          discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5)
          encoder_optimizer=...
          )

      # Train estimator.
      gan_estimator.train(train_input_fn, steps)

      # Evaluate resulting estimator.
      gan_estimator.evaluate(eval_input_fn)

      # Generate samples from generator.
      predictions = np.array([
          x for x in gan_estimator.predict(predict_input_fn)])
  ```
  """

    def __init__(self,
                 model_dir=None,
                 generator_fn=None,
                 discriminator_fn=None,
                 encoder_fn=None,
                 generator_loss_fn=None,
                 encoder_loss_fn=None,
                 discriminator_loss_fn=None,
                 generator_optimizer=None,
                 encoder_optimizer=None,
                 discriminator_optimizer=None,
                 get_hooks_fn=None,
                 get_eval_metric_ops_fn=None,
                 add_summaries=None,
                 use_loss_summaries=True,
                 config=None,
                 params=None,
                 warm_start_from=None,
                 is_chief=True):
        # TODO (davit): write docstring
        """Initializes a GANEstimator instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      generator_fn: A python function that takes a Tensor, Tensor list, or
        Tensor dictionary as inputs and returns the outputs of the GAN
        generator. See `TF-GAN` for more details and examples. Additionally, if
        it has an argument called `mode`, the Estimator's `mode` will be passed
        in (ex TRAIN, EVAL, PREDICT). This is useful for things like batch
        normalization.
      discriminator_fn: A python function that takes the output of
        `generator_fn` or real data in the GAN setup, and `generator_inputs`.
        Outputs a Tensor in the range [-inf, inf]. See `TF-GAN` for more details
        and examples.
      generator_loss_fn: The loss function on the generator. Takes a `GANModel`
        tuple.
      discriminator_loss_fn: The loss function on the discriminator. Takes a
        `GANModel` tuple.
      generator_optimizer: The optimizer for generator updates, or a function
        that takes no arguments and returns an optimizer. This function will
        be called when the default graph is the `GANEstimator`'s graph, so
        utilities like `tf.train.get_or_create_global_step` will
        work.
      discriminator_optimizer: Same as `generator_optimizer`, but for the
        discriminator updates.
      get_hooks_fn: A function that takes a `GANTrainOps` tuple and returns a
        list of hooks. These hooks are run on the generator and discriminator
        train ops, and can be used to implement the GAN training scheme.
        Defaults to `train.get_sequential_train_hooks()`.
      get_eval_metric_ops_fn: A function that takes a `GANModel`, and returns a
        dict of metric results keyed by name. The output of this function is
        passed into `tf.estimator.EstimatorSpec` during evaluation.
      add_summaries: `None`, a single `SummaryType`, or a list of `SummaryType`.
      use_loss_summaries: If `True`, add loss summaries. If `False`, does not.
        If `None`, uses defaults.
      config: `RunConfig` object to configure the runtime settings.
      params: Optional `dict` of hyperparameters.  Will receive what is passed
        to Estimator in `params` parameter. This allows to configure Estimators
        from hyper parameter tuning. If any `params` are args to TF-GAN's
        `gan_loss`, they will be passed to `gan_loss` during training and
        evaluation.
      warm_start_from: A filepath to a checkpoint or saved model, or a
        WarmStartSettings object to configure initialization.
      is_chief: Whether or not this Estimator is running on a chief or worker.
        Needs to be set appropriately if using SyncReplicasOptimizers.

    Raises:
      ValueError: If loss functions aren't callable.
      ValueError: If `use_loss_summaries` isn't boolean or `None`.
      ValueError: If `get_hooks_fn` isn't callable or `None`.
    """
        _validate_input_args(generator_loss_fn, encoder_loss_fn, discriminator_loss_fn,
                             use_loss_summaries, get_hooks_fn)
        optimizers = Optimizers(generator_optimizer, discriminator_optimizer, encoder_optimizer)

        def _model_fn(features, labels, mode, params):
            """GANEstimator model function."""
            if mode not in [
                tf.estimator.ModeKeys.TRAIN,
                tf.estimator.ModeKeys.EVAL,
                tf.estimator.ModeKeys.PREDICT
            ]:
                raise ValueError('Mode not recognized: %s' % mode)

            #bs = params['batch_size']
            #nd = params['noise_dims']
            gp_weight = params['gp_weight']

            # rename inputs for clarity
            if type(features) == dict:
                real_data = features['real_input']
                # generator_inputs = features['g_input']
            else:
                real_data = features

                # # # g_input = tf.Variable(tf.zeros([self.batch_size, self.input_dims]))
                # # g_input = tf.compat.v1.get_variable(
                # #     name='g_input',
                # #     initializer=tf.random.normal([params['batch_size'], params['noise_dims']], mean=0, stddev=1,
                # #                                  name="G_noise_input")
                # # )
                # # TODO (davit): this is very ugly. please move this to provide_data_from_tfrecord_files and then
                # #   it will be all dict
                # if 'd1_net' in params:
                #     if params['d1_net']:
                #         g_input = tf.compat.v1.get_variable(
                #             name='g_input',
                #             initializer=tf.random.normal([params['batch_size'], params['time_steps'], params['feature_dim']],
                #                                          mean=0, stddev=1, dtype=tf.float32, name="G_noise_input"))
                #         # for tensorboard
                #         g_update_placeholder = tf.compat.v1.placeholder(tf.float32,
                #                             shape=(params['batch_size'], params['time_steps'], params['feature_dim']),
                #                                                         name="g_update_placeholder")
                #     else:
                #         g_input = tf.compat.v1.get_variable(
                #             name='g_input',
                #             initializer=tf.random.normal([params['batch_size'], params['noise_dims']], mean=0, stddev=1,
                #                                          name="G_noise_input")
                #         )
                #         # for tensorboard
                #         g_update_placeholder = tf.compat.v1.placeholder(tf.float32,
                #                                                         shape=(params['batch_size'], params['noise_dims']),
                #                                                         name="g_update_placeholder")
                #
                # else:
                #     g_input = tf.compat.v1.get_variable(
                #         name='g_input',
                #         initializer=tf.random.normal([params['batch_size'], params['noise_dims']], mean=0, stddev=1,
                #                                      name="G_noise_input")
                #     )
                #     # for tensorboard
                #     g_update_placeholder = tf.compat.v1.placeholder(tf.float32,
                #                                                     shape=(params['batch_size'], params['noise_dims']),
                #                                                     name="g_update_placeholder")

            g_input = tf.compat.v1.get_variable(
                name='g_input',
                initializer=tf.random.normal([params['batch_size'], params['noise_dims']], mean=0, stddev=1,
                                             name="G_noise_input")
                ,
                synchronization=tf.VariableSynchronization.AUTO,
                aggregation=tf.compat.v1.VariableAggregation.MEAN
            )
            # for tensorboard
            g_update_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                            shape=(params['batch_size'], params['noise_dims']),
                                                            name="g_update_placeholder")

            #----------------------------------------------------------------------------------------------------------#

            generator_inputs = g_input.read_value()

            tf.compat.v1.assign(g_input, g_update_placeholder, use_locking=None, name="g_update")
            tf.compat.v1.summary.histogram('g_input', g_input)

            real_input = tf.compat.v1.identity(real_data, name="real_input")
            tf.compat.v1.summary.histogram('real_input', real_input)

            # generator_inputs = (tf.data.Dataset.from_tensors(0).repeat().map(lambda _: tf.random.normal([bs, nd])))
            # generator_inputs = tf.random_normal([bs, nd], mean=0, stddev=1, name="G_noise_input")

            print(">>>>>>>>>>>>>>>>>>>>>>>>")
            print(features)
            print(labels)
            print(generator_inputs)
            print(">>>>>>>>>>>>>>>>>>>>>>>>")

            # Make GANModel, which encapsulates the GAN model architectures.
            gan_model = get_gan_model(mode,
                                      generator_fn,
                                      discriminator_fn,
                                      encoder_fn,
                                      real_data,
                                      labels,
                                      generator_inputs,
                                      add_summaries)

            # Make GANLoss, which encapsulates the losses.
            if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
                gan_loss_kwargs = extract_gan_loss_args_from_params(params) or {}
                gan_loss = tfgan_train.gan_enc_loss(
                    gan_model,
                    generator_loss_fn,
                    encoder_loss_fn,
                    discriminator_loss_fn,
                    gradient_penalty_weight=gp_weight,
                    add_summaries=use_loss_summaries,
                    **gan_loss_kwargs)

            # Make the EstimatorSpec, which incorporates the GANModel, losses, eval
            # metrics, and optimizers (if required).
            if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
                estimator_spec = get_train_estimator_spec(
                    gan_model, gan_loss, optimizers, get_hooks_fn, is_chief=is_chief)

            elif mode == tf.compat.v1.estimator.ModeKeys.EVAL:
                estimator_spec = get_eval_estimator_spec(
                    gan_model, gan_loss, labels, get_eval_metric_ops_fn=get_eval_metric_ops_fn)
                   #gan_model, gan_loss, labels, params, get_eval_metric_ops_fn=get_eval_metric_ops_fn)

            else:  # tf.estimator.ModeKeys.PREDICT
                estimator_spec = get_predict_estimator_spec(gan_model)
                #estimator_spec = get_predict_estimator_spec(gan_model, params=params)

            return estimator_spec

        super(GANEncAnomalyEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config, params=params,
            warm_start_from=warm_start_from)


def _validate_input_args(generator_loss_fn, encoder_loss_fn, discriminator_loss_fn,
                         use_loss_summaries, get_hooks_fn):
    if not callable(generator_loss_fn):
        raise ValueError('generator_loss_fn must be callable.')
    if not callable(encoder_loss_fn):
        raise ValueError('encoder_loss_fn must be callable.')
    if not callable(discriminator_loss_fn):
        raise ValueError('discriminator_loss_fn must be callable.')
    if use_loss_summaries not in [True, False, None]:
        raise ValueError('use_loss_summaries must be True, False or None.')
    if get_hooks_fn is not None and not callable(get_hooks_fn):
        raise TypeError('get_hooks_fn must be callable.')



def get_gan_model(mode,
                  generator_fn,
                  discriminator_fn,
                  encoder_fn,
                  real_data,
                  labels,
                  generator_inputs,
                  add_summaries,
                  generator_scope='Generator',
                  discriminator_scope='Discriminator',
                  encoder_scope='Encoder'):
    """Makes the GANModel tuple, which encapsulates the GAN model architecture."""
    if mode == tf.estimator.ModeKeys.PREDICT:
        # FIXME (davit): decide here about real_data and labels
        # if real_data is not None:
        #  raise ValueError('`labels` must be `None` when mode is `predict`. '
        #                   'Instead, found %s' % real_data)

        gan_model = make_prediction_gan_model(generator_fn, discriminator_fn,
                                    encoder_fn, real_data,
                                    generator_inputs, generator_scope,
                                    discriminator_scope, encoder_scope,
                                    add_summaries, mode)

    else:  # tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.EVAL
        gan_model = _make_gan_model(generator_fn,
                                    discriminator_fn,
                                    encoder_fn,
                                    real_data,
                                    labels,
                                    generator_inputs,
                                    generator_scope,
                                    discriminator_scope,
                                    encoder_scope,
                                    add_summaries, mode)

    return gan_model


def _make_gan_model(generator_fn,
                    discriminator_fn,
                    encoder_fn,
                    real_data,
                    labels,
                    generator_inputs,
                    generator_scope,
                    discriminator_scope,
                    encoder_scope,
                    add_summaries,
                    mode):
    """Construct a `GANModel`, and optionally pass in `mode`."""
    # If network functions have an argument `mode`, pass mode to it.
    if 'mode' in inspect.getargspec(generator_fn).args:
        generator_fn = functools.partial(generator_fn, mode=mode)
    if 'mode' in inspect.getargspec(discriminator_fn).args:
        discriminator_fn = functools.partial(discriminator_fn, mode=mode)
    if 'mode' in inspect.getargspec(encoder_fn).args:
        encoder_fn = functools.partial(encoder_fn, mode=mode)


    gan_model = tfgan_train.gan_enc_anomaly_model(
        generator_fn,
        discriminator_fn,
        encoder_fn,
        real_data,
        labels,
        generator_inputs,
        generator_scope=generator_scope,
        discriminator_scope=discriminator_scope,
        encoder_scope=encoder_scope,
        check_shapes=False)

    if add_summaries:
        # FIXME (davit)
        pass
        # # if not isinstance(add_summaries, (tuple, list)):
        # #     add_summaries = [add_summaries]
        # with tf.compat.v1.name_scope(''):  # Clear name scope.
        #     for summary_type in add_summaries:
        #         summary_type_map[summary_type](gan_model)
        #
        #     # summary_type_map = {SummaryType.VARIABLES: tfgan_summaries.add_gan_anomaly_model_summaries}

    return gan_model



def make_prediction_gan_model(generator_fn, discriminator_fn,
                    encoder_fn,
                    real_data,
                    generator_inputs, generator_scope,
                    discriminator_scope, encoder_scope,
                    add_summaries, mode):
    # FIXME (davit): decide here about real_data and labels
    #   please look at https://arxiv.org/pdf/1905.11034.pdf
    """Make a `GANModel` from just the generator."""
    # If `generator_fn` has an argument `mode`, pass mode to it.

    labels = None
    gan_model = _make_gan_model(generator_fn,
                                discriminator_fn,
                                encoder_fn,
                                real_data,
                                labels,
                                generator_inputs, generator_scope,
                                discriminator_scope, encoder_scope,
                                add_summaries, mode
                                )

    return tfgan_tuples.GANEncoderAnomalyModel(
        generator_inputs=None,
        generated_data=None,
        generator_variables=None,
        generator_scope=None,
        generator_fn=None,
        real_data=None,
        discriminator_real_outputs=None,
        discriminator_gen_outputs=None,
        discriminator_variables=None,
        discriminator_scope=None,
        discriminator_fn=None,
        generator_reconstracted_data=None,
        encoder_scope=None,
        encoder_fn=None,
        encoded_generated_data=None,
        encoder_variables=None,
        anomaly_score=gan_model.anomaly_score,
        gen_rec_loss=None,
        real_to_orig_dist=None,
        gen_rec_loss_predict=gan_model.gen_rec_loss_predict,
        real_to_orig_dist_predict=gan_model.real_to_orig_dist_predict
        # ano_anomaly_score,
        # ano_gen_rec_loss,
        # ano_real_to_orig_dist,
        # ben_anomaly_score,
        # ben_gen_rec_loss,
        # ben_real_to_orig_dist
    )


def get_eval_estimator_spec(gan_model, gan_loss, labels, get_eval_metric_ops_fn=None):
#def get_eval_estimator_spec(gan_model, gan_loss, labels, params, get_eval_metric_ops_fn=None):
    """Return an EstimatorSpec for the eval case."""

    with tf.compat.v1.name_scope(name='metrics'):
        eval_metric_ops = {
            'generator_loss':
                tf.compat.v1.metrics.mean(gan_loss.generator_loss),
            'discriminator_loss':
                tf.compat.v1.metrics.mean(gan_loss.discriminator_loss),
            'encoder_loss':
                tf.compat.v1.metrics.mean(gan_loss.encoder_loss),
            # 'ano_anomaly_score':
            #     tf.compat.v1.metrics.mean(gan_model.ano_anomaly_score),
            # 'ano_gen_rec_loss':
            #     tf.compat.v1.metrics.mean(gan_model.ano_gen_rec_loss),
            # 'ano_real_to_orig_dist':
            #     tf.compat.v1.metrics.mean(gan_model.ano_real_to_orig_dist),
            # 'ben_anomaly_score':
            #     tf.compat.v1.metrics.mean(gan_model.ben_anomaly_score),
            # 'ben_gen_rec_loss':
            #     tf.compat.v1.metrics.mean(gan_model.ben_gen_rec_loss),
            # 'ben_real_to_orig_dist':
            #     tf.compat.v1.metrics.mean(gan_model.ben_real_to_orig_dist),
            #'recall':
            #    tf.compat.v1.metrics.recall(labels, gan_model.anomaly_score > params['anomaly_threshold']),
            #'precision':
            #    tf.compat.v1.metrics.precision(labels, gan_model.anomaly_score > params['anomaly_threshold']),
            #'accuracy':
            #    tf.compat.v1.metrics.accuracy(labels, gan_model.anomaly_score > params['anomaly_threshold'])
        }

        # eval_metric_ops.update(_get_metrics(gan_model.anomaly_score, labels))

        if get_eval_metric_ops_fn is not None:
            custom_eval_metric_ops = get_eval_metric_ops_fn
            if not isinstance(custom_eval_metric_ops, dict):
                raise TypeError('get_eval_metric_ops_fn must return a dict, '
                                'received: {}'.format(custom_eval_metric_ops))
            eval_metric_ops.update(custom_eval_metric_ops)
    return tf.compat.v1.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        predictions=gan_model.anomaly_score,
        loss=gan_loss.discriminator_loss,
        eval_metric_ops=eval_metric_ops
        # training_hooks=[ModelStats(gan_model.generator_inputs, gan_model.generated_data,
        #                            gan_model.generator_reconstracted_data, gan_model.encoded_generated_data,labels)]
    )


# TODO (davit): prediction part is work in progress
#   please look at https://arxiv.org/pdf/1905.11034.pdf
def get_predict_estimator_spec(gan_model):
#def get_predict_estimator_spec(gan_model, params):
    # FIXME (davit): decide here about real_data and labels
    """Return an EstimatorSpec for the predict case."""
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                      predictions=gan_model.anomaly_score,
                                      #predictions=gan_model.anomaly_score > params['anomaly_threshold'])
                                      export_outputs= {
                                          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                              tf.estimator.export.PredictOutput(outputs={'gen_rec_loss_predict': gan_model.gen_rec_loss_predict}),
                                          'head-2': tf.estimator.export.PredictOutput(outputs={'real_to_orig_dist_predict': gan_model.real_to_orig_dist_predict})
                                      }
    )




def _maybe_construct_optimizers(optimizers):
    g_callable = callable(optimizers.gopt)
    gopt = optimizers.gopt() if g_callable else optimizers.gopt
    e_callable = callable(optimizers.eopt)
    eopt = optimizers.eopt() if e_callable else optimizers.eopt
    d_callable = callable(optimizers.dopt)
    dopt = optimizers.dopt() if d_callable else optimizers.dopt

    return Optimizers(gopt, dopt, eopt)


def get_train_estimator_spec(
        gan_model, gan_loss, optimizers,
        get_hooks_fn, train_op_fn=tfgan_train.gan_enc_train_ops, is_chief=True):
    """Return an EstimatorSpec for the train case."""
    get_hooks_fn = get_hooks_fn or tfgan_train.get_sequential_train_hooks()
    optimizers = _maybe_construct_optimizers(optimizers)

    train_ops = train_op_fn(gan_model, gan_loss, optimizers.gopt,
                            optimizers.dopt, optimizers.eopt, is_chief=is_chief)
    training_hooks = get_hooks_fn(train_ops)
    return tf.compat.v1.estimator.EstimatorSpec(
        loss=gan_loss.discriminator_loss, #TODO (davit): only discriminator loss?
        mode=tf.estimator.ModeKeys.TRAIN,
        train_op=train_ops.global_step_inc_op,
        training_hooks=training_hooks)


def extract_gan_loss_args_from_params(params):
    """Returns a dictionary with values for `gan_loss`."""
    gan_loss_arg_names = inspect.getargspec(tfgan_train.gan_loss).args

    # Remove args that you can't adjust via params, and fail if they're present.
    for forbidden_symbol in [
        'model', 'generator_loss_fn', 'discriminator_loss_fn', 'add_summaries']:
        gan_loss_arg_names.remove(forbidden_symbol)
        if forbidden_symbol in params:
            raise ValueError('`%s` is not allowed in params.')

    gan_loss_args = {k: params[k] for k in gan_loss_arg_names if k in params}

    return gan_loss_args

