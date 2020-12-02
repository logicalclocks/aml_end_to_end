import numpy as np
import tensorflow as tf


def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    # TODO (davit): provide ref https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """

    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank,
    # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None

    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        #        assert X.shape[0] == Y.shape[0]
        #        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)

        # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]

        d_feature = {}
        d_feature['features'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['target'] = dtype_feature_y(y)

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))
