import tensorflow as tf
from tensorflow.python.tools import saved_model_utils

import os
import uuid

from hops import model as hops_model

export_path = os.getcwd() + '/model-' + str(uuid.uuid4())

def export_saved_model(estimator, input_feature_dim, metrics, model_name):

    """
    :param estimator: gan_estimator object
    :param features:  a dict of string to Tensor.
    :param export_dir_path: saved model path
    """

    feature_spec = {
        'real_input': tf.placeholder(tf.float32, shape=(None, input_feature_dim), name='real_input')
    }
    # Define the input receiver for the raw tensors
    def serving_input_receiver_fn():
        return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()

    export_tmp_path = os.getcwd() + '/model-tmp' + str(uuid.uuid4())
    export_path = os.getcwd() + '/model-' + str(uuid.uuid4())

    estimator.export_savedmodel(
        export_dir_base=export_tmp_path,
        serving_input_receiver_fn=serving_input_receiver_fn
    )
    covert_to_saved_model(model_path_dir=export_tmp_path, export_path=export_path)
    hops_model.export(export_path, model_name, metrics=metrics)

def covert_to_saved_model(model_path_dir, export_path):

    if os.path.isdir(model_path_dir):
        if not model_path_dir.endswith("/"):
            model_path = model_path_dir + "/"
    for filename in os.listdir(model_path):
        if os.path.isdir(model_path + filename):
            model_path = model_path + filename

    meta_graph_def = get_serving_meta_graph_def(model_path)
    signature_def = [meta_graph_def.signature_def[key] for key in [
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, 'head-2']]
    outputs = [v.name for k in signature_def for v in k.outputs.values()]
    output_names = [node for node in outputs]

    print("--------------output_names ----------------")
    print(output_names)
    print("------------------------------")

    inputs = [v.name for k in signature_def for v in k.inputs.values()]
    input_names = [node for node in inputs]

    print("--------------input_names ----------------")
    print(input_names)
    print("------------------------------")

    with tf.Graph().as_default():
        with tf.Session() as session:
            # reload the model to add the default signature
            tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_path)

            # build tensors info for the inputs & the outputs into the signature def
            signature_def = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'real_input': tf.saved_model.utils.build_tensor_info(tf.get_default_graph().get_tensor_by_name(input_names[0]))
                    },
                    outputs={
                        'gen_rec_loss_predict': tf.saved_model.utils.build_tensor_info(
                            # cast the ExpandDims tensors into a float instead of an int64
                            tf.reshape(
                                tf.cast(
                                    tf.get_default_graph().get_tensor_by_name(
                                        output_names[0]),
                                    dtype=tf.float32
                                ), [-1])
                        ),
                        'real_to_orig_dist_predict': tf.saved_model.utils.build_tensor_info(
                            # cast the ExpandDims tensors into a float instead of an int64
                            tf.reshape(
                                tf.cast(
                                    tf.get_default_graph().get_tensor_by_name(
                                        output_names[1]),
                                    dtype=tf.float32
                                ), [-1])
                        ),

                    },

                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )


            # save the model with proper format
            print('Exporting trained model to: {}'.format(export_path))
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            builder.add_meta_graph_and_variables (
                session, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY : signature_def
                },
            )
            # save the model with a default signature
            builder.save()

def get_serving_meta_graph_def(savedmodel_dir):
    """Extract the SERVING MetaGraphDef from a SavedModel directory.

    Args:
      savedmodel_dir: the string path to the directory containing the .pb
        and variables for a SavedModel. This is equivalent to the subdirectory
        that is created under the directory specified by --export_dir when
        running an Official Model.

    Returns:
      MetaGraphDef that should be used for tag_constants.SERVING mode.

    Raises:
      ValueError: if a MetaGraphDef matching tag_constants.SERVING is not found.
    """
    # We only care about the serving graph def


    tag_set = set([tf.saved_model.tag_constants.SERVING])
    #print("tag_set: ",tag_set)
    serving_graph_def = None
    saved_model = saved_model_utils.read_saved_model(savedmodel_dir)
    for meta_graph_def in saved_model.meta_graphs:
        #print("meta_graph_def: ",meta_graph_def)
        if set(meta_graph_def.meta_info_def.tags) == tag_set:
            serving_graph_def = meta_graph_def
    if not serving_graph_def:
        raise ValueError("No MetaGraphDef found for tag_constants.SERVING. "
                         "Please make sure the SavedModel includes a SERVING def.")

    return serving_graph_def


# def _get_model_full_path(model_base_path, version=None):
#     # https://github.com/yupbank/tf-spark-serving
#     """
#     Get the full path for a model.
#     If version arg is not specified, we return the full path for the latest version.
#     If version arg is specified, we verify that the version actually exists.
#     """
#     versions = {}
#     DIGIT_PATTERN = '[0-9]*'
#     model_base_path = model_base_path.rstrip('/')
#     for path in tf.gfile.Glob(os.path.join(model_base_path, DIGIT_PATTERN)):
#         _, v = path.rsplit('/', 1)
#         versions[int(v)] = path
#
#     if version is None:
#         version = max(versions.keys())
#     else:
#         if not str(version).isdigit():
#             raise Exception('Only int/long version is allowed')
#         version = int(version)
#
#     if version not in versions:
#         raise Exception('Given version %s not found' % version)
#
#     return versions[version]
#
# def load_model(model_base_path=None, model_version=None, model_full_path=None, signature_def_key=None):
#     # https://github.com/yupbank/tf-spark-serving
#     """
#     Load the model_version/latest savedModel from model_base_path.
#     We use the default signature_def_key 'default_serving' by default.
#     """
#     assert model_base_path or model_full_path, "You have to specify either model_base_path or model_full_path"
#     signature_def_key = signature_def_key or tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
#     model_full_path = model_full_path or _get_model_full_path(
#         model_base_path, model_version)
#     return tf.contrib.predictor.from_saved_model(model_full_path, signature_def_key)
#
# def rename_by_mapping(df, tensor_mapping, reverse=False):
#     # https://github.com/yupbank/tf-spark-serving
#     for name, tensor in tensor_mapping.items():
#         renames = name, tensor.op.name
#         if reverse:
#             renames = reversed(renames)
#         df = df.withColumnRenamed(*renames)
#     return df
#
# def tf_serving_with_dataframe(df, model_base_path, model_version=None):
#     # https://github.com/yupbank/tf-spark-serving
#     """
#     :param df: spark dataframe, batch input for the model
#     :param model_base_path: str, tensorflow saved Model model base path
#     :param model_version: int, tensorflow saved Model model version, default None
#     :return: spark dataframe, with predicted result.
#     """
#     import tensorframes as tfs
#
#     g, feed_tensors, fetch_tensors = load_model(model_base_path, model_version)
#     with g.as_default():
#         df = rename_by_mapping(df, feed_tensors)
#         df = tfs.analyze(df)
#         df = tfs.map_blocks(fetch_tensors.values(), df)
#         df = rename_by_mapping(df, feed_tensors, reverse=True)
#         return rename_by_mapping(df, fetch_tensors, reverse=True)
