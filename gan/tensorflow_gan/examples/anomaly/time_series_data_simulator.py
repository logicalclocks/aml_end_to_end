import numpy as np
import pandas as pd

from absl import app
from absl import flags

flags.DEFINE_integer('time_steps', 365, 'Time steps per observation.')
flags.DEFINE_integer('normal_sample_size', 1000, 'Sample size of normal cases.')
flags.DEFINE_integer('ano_sample_size', 0, 'Sample size of anomalous cases.')
flags.DEFINE_string('data_file_path', '/tmp/test.csv', 'Path to csv file for generated dataset.')
flags.DEFINE_string('labels_file_path', '/tmp/labels.csv', 'Path to csv file for labels of generated dataset.')

FLAGS = flags.FLAGS

# TODO (davit): provide reference if this functions
#  https://github.com/KDD-OpenSource/agots/blob/master/agots/multivariate_generators/multivariate_data_generator.py
class MultivariateOutlierGenerator:
    def __init__(self, timestamps):
        self.timestamps = timestamps

    def add_outliers(self, timeseries):
        return NotImplementedError


class MultivariateExtremeOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor=8):
        self.timestamps = [] if timestamps is None else list(sum(timestamps, ()))
        self.factor = factor

    def get_value(self, current_timestamp, timeseries):
        if current_timestamp in self.timestamps:
            local_std = timeseries.iloc[max(0, current_timestamp - 10):current_timestamp + 10].std()
            return np.random.choice([-1, 1]) * self.factor * local_std
        else:
            return 0

    def add_outliers(self, timeseries):
        additional_values = []
        for timestamp_index in range(len(timeseries)):
            additional_values.append(self.get_value(timestamp_index, timeseries))
        return additional_values


class MultivariateShiftOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor=8):
        timestamps = timestamps or []
        self.timestamps = timestamps
        self.factor = factor

    def add_outliers(self, timeseries):
        additional_values = np.zeros(timeseries.size)
        for start, end in self.timestamps:
            local_std = timeseries.iloc[max(0, start - 10):end + 10].std()
            additional_values[list(range(start, end))] += np.random.choice([-1, 1]) * self.factor * local_std
        return additional_values


class MultivariateTrendOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor=8):
        self.timestamps = timestamps or []
        self.factor = factor / 10  # Adjust for trend

    def add_outliers(self, timeseries):
        additional_values = np.zeros(timeseries.size)
        for start, end in self.timestamps:
            slope = np.random.choice([-1, 1]) * self.factor * np.arange(end - start)
            additional_values[list(range(start, end))] += slope
            additional_values[end:] += slope[-1]
        return additional_values


class MultivariateVarianceOutlierGenerator(MultivariateOutlierGenerator):
    def __init__(self, timestamps=None, factor=8):
        self.timestamps = timestamps or []
        self.factor = factor

    def add_outliers(self, timeseries):
        additional_values = np.zeros(timeseries.size)
        for start, end in self.timestamps:
            difference = np.diff(timeseries[start - 1:end]) if start > 0 \
                else np.insert(np.diff(timeseries[start:end]), 0, 0)
            additional_values[list(range(start, end))] += (self.factor - 1) * difference
        return additional_values


INITIAL_VALUE_MIN = 0
INITIAL_VALUE_MAX = 1


class MultivariateDataGenerator:
    def __init__(self, stream_length, n, k, shift_config=None, behavior=None, behavior_config=None):
        """Create multivariate time series using outlier generators
        :param stream_length: number of values in each time series
        :param n: number of time series at all
        :param k: number of time series that should correlate. If all should correlate with the first
        one, set k=n.
        :param shift_config: dictionary from index of the time series to how much it should be displaced in time (>=0)
        """

        if not shift_config:
            self.shift_config = {}
            self.max_shift = 0
        else:
            self.shift_config = shift_config
            self.max_shift = max(list(self.shift_config.values()))
        self.behavior = behavior
        self.behavior_config = behavior_config if behavior_config is not None else {}

        self.STREAM_LENGTH = stream_length
        self.N = n
        self.K = k
        self.data = pd.DataFrame()
        self.outlier_data = pd.DataFrame()

        assert self.STREAM_LENGTH > 0, 'stream_length must at least be 1'
        assert self.N > 0, 'n must at least be 1'
        assert self.K >= 0, 'k must at least be 0'
        assert self.K <= self.N, 'k must be less than or equal to n'
        assert 0 not in self.shift_config.keys(), 'The origin time series cannot be shifted in time'

        if k == 0:  # There is no difference between k=0 and k=1.
            self.K = 1

    def generate_baseline(self, correlation_min=0.9, correlation_max=0.7, initial_value_min=INITIAL_VALUE_MIN,
                          initial_value_max=INITIAL_VALUE_MAX):
        """
        Generate the multivariate data frame
        :param correlation_min: how much the k columns should at least correlate with the first one
        :param correlation_max: how much the n-k-1 columns should at max correlate with the first one
        :param initial_value_min: minimal possible value of the first entry of the time series
        :param initial_value_max: maximal possible value of the first entry of the time series
        :return: a DataFrame with columns ['timestamp', 'x0', ... 'xn-1'] in which the first column
        is the original time series. The following k columns correlate at least with correlation_min
        with it and the remaining n-k columns correlate at max correlation_max with it.
        """

        df = self.init_dataframe(self.N)
        # Create k correlating time series
        df = self.create_correlating_time_series(self.K, correlation_min, df, initial_value_min, initial_value_max)

        # Create the remaining n - k time series randomly
        df = self.create_not_correlating_time_series(self.K, self.N, correlation_max, df, initial_value_min,
                                                     initial_value_max)

        # Perform the shifts: currently all time series have n+max_shift elements
        # Each one should start at index max_shift - own_shift such that the padded measurements of a time series before
        # the origin time series starts descend from a self-correlating distribution
        for k, column_name in enumerate(df.columns):
            own_shift = 0 if k not in self.shift_config.keys() else self.shift_config[k]
            df[column_name] = df[column_name].shift(own_shift)

        df.dropna(axis=0, inplace=True)
        df.reset_index(inplace=True, drop=True)

        assert not df.isnull().values.any(), 'There is at least one NaN in the generated DataFrame'
        self.data = df
        return self.data

    def init_dataframe(self, number_time_series):
        columns = ['timestamp']
        for value_column_index in range(number_time_series):
            columns.append('x{}'.format(value_column_index))
        df = pd.DataFrame(columns=columns)
        return df

    def create_basic_time_series(self, df, initial_value_min=INITIAL_VALUE_MIN, initial_value_max=INITIAL_VALUE_MAX):
        if initial_value_min != initial_value_max:
            start = np.random.randint(initial_value_min, initial_value_max)
        else:
            start = initial_value_min

        if self.behavior is not None:
            behavior_generator = self.behavior(**self.behavior_config)

        # Create basic time series
        x = [start]
        timestamps = [0]
        for i in range(1, self.STREAM_LENGTH + self.max_shift):
            timestamps.append(i)
            value = x[i - 1] + np.random.normal(0, 1)
            if self.behavior is not None:
                value += next(behavior_generator)
            x.append(value)
        df['x0'] = x
        df['timestamp'] = timestamps
        df.set_index('timestamp', inplace=True)
        return df

    def create_correlating_time_series(self, number_time_series, correlation_min, df,
                                       initial_value_min=INITIAL_VALUE_MIN,
                                       initial_value_max=INITIAL_VALUE_MAX):
        # First time series
        df = self.create_basic_time_series(df=df, initial_value_min=initial_value_min,
                                           initial_value_max=initial_value_max)
        origin_offset = df.iloc[0, 0]

        # number_time_series time series which are correlating
        for index_correlating in range(1, number_time_series):
            while True:
                x = [0]
                if initial_value_min != initial_value_max:
                    offset = np.random.randint(initial_value_min, initial_value_max)
                else:
                    offset = initial_value_min
                for index_timeseries_length in range(self.STREAM_LENGTH - 1 + self.max_shift):
                    # Take 50% of time series 0 and add 50% randomness
                    original_value = df.iloc[index_timeseries_length, 0] - origin_offset
                    x.append(0.5 * original_value + 0.5 * (np.random.random() - 0.5))
                df['x' + str(index_correlating)] = x
                df['x' + str(index_correlating)] += offset
                if abs(df.corr().iloc[0, index_correlating]) >= correlation_min:
                    break
            assert (len(df) == self.STREAM_LENGTH + self.max_shift)
        return df

    def create_not_correlating_time_series(self, k, n, correlation_max, df, initial_value_min=INITIAL_VALUE_MIN,
                                           initial_value_max=INITIAL_VALUE_MAX):
        for index_not_correlation in range(k, n):
            if self.behavior is not None:
                behavior_generator = self.behavior(**self.behavior_config)
            while True:
                if initial_value_min != initial_value_max:
                    x = [np.random.randint(initial_value_min, initial_value_max)]
                else:
                    x = [initial_value_min]
                for index_timeseries_length in range(self.STREAM_LENGTH - 1 + self.max_shift):
                    value = x[index_timeseries_length] + (np.random.random() - 0.5)
                    if self.behavior is not None:
                        value += next(behavior_generator)
                    x.append(value)
                df['x' + str(index_not_correlation)] = x
                if abs(df.corr().iloc[0, index_not_correlation]) <= correlation_max:
                    break
            assert (len(df) == self.STREAM_LENGTH + self.max_shift)
        return df

    def add_outliers(self, config):
        """Adds outliers based on the given configuration to the base line
         :param config: Configuration file for the outlier addition e.g.
         {'extreme': [{'n': 0, 'timestamps': [(3,)]}],
          'shift':   [{'n': 3, 'timestamps': [(4,10)]}]}
          would add an extreme outlier to time series 0 at timestamp 3 and a base shift
          to time series 3 between timestamps 4 and 10
         :return:
         """
        OUTLIER_GENERATORS = {'extreme': MultivariateExtremeOutlierGenerator,
                              'shift': MultivariateShiftOutlierGenerator,
                              'trend': MultivariateTrendOutlierGenerator,
                              'variance': MultivariateVarianceOutlierGenerator}

        generator_keys = []

        # Validate the input
        for outlier_key, outlier_generator_config in config.items():
            assert outlier_key in OUTLIER_GENERATORS, 'outlier_key must be one of {} but was'.format(OUTLIER_GENERATORS,
                                                                                                     outlier_key)
            generator_keys.append(outlier_key)
            for outlier_timeseries_config in outlier_generator_config:
                n, timestamps = outlier_timeseries_config['n'], outlier_timeseries_config['timestamps']
                assert n in range(self.N), 'n must be between 0 and {} but was {}'.format(self.N - 1, n)
                for timestamp in list(sum(timestamps, ())):
                    assert timestamp in range(
                        self.STREAM_LENGTH), 'timestamp must be between 0 and {} but was {}'.format(self.STREAM_LENGTH,
                                                                                                    timestamp)

        df = self.data
        if self.data.shape == (0, 0):
            raise Exception('You have to first compute a base line by invoking generate_baseline()')
        for generator_key in generator_keys:
            for outlier_timeseries_config in config[generator_key]:
                n, timestamps = outlier_timeseries_config['n'], outlier_timeseries_config['timestamps']
                generator_args = dict(
                    [(k, v) for k, v in outlier_timeseries_config.items() if k not in ['n', 'timestamps']])
                generator = OUTLIER_GENERATORS[generator_key](timestamps=timestamps, **generator_args)
                df[df.columns[n]] += generator.add_outliers(self.data[self.data.columns[n]])

        assert not df.isnull().values.any(), 'There is at least one NaN in the generated DataFrame'
        self.outlier_data = df
        return df


def multi_var_ben_ano_gen(time_steps, sample_size):
    baseline = MultivariateDataGenerator(stream_length=time_steps, n=sample_size, k=0)
    baseline.generate_baseline()
    # would add an extreme outlier to time series 0 at timestamp 3 and a base shift
    #           to time series 3 between timestamps 4 and 10

    shift_start_index = np.random.randint(time_steps - int(time_steps / 4), size=1)[0]
    shift_end_index = \
    np.random.randint(shift_start_index + 1, shift_start_index + int(int(time_steps / 4) / 2), size=1)[0]

    int_to_ano_type = {1: 'extreme',
                       2: 'shift',
                       3: 'variance',
                       4: 'trend'}

    ano_type_key = np.random.randint(1, 4, size=1)[0]

    ano_data = baseline.add_outliers(
        {int_to_ano_type[ano_type_key]: [{'n': 0, 'timestamps': [(shift_start_index, shift_end_index)]}]}
    )

    normal_baseline = MultivariateDataGenerator(stream_length=time_steps, n=sample_size, k=0)
    normal_data = normal_baseline.generate_baseline()
    return normal_data.transpose().values, ano_data.transpose().values


#    return normal_data.transpose().values.reshape([-1,time_steps,1]), ano_data.transpose().values.reshape([-1,time_steps,1])


def generate_time_series(time_steps, normal_sample_size, ano_sample_size=0):
    x, _ = multi_var_ben_ano_gen(time_steps=time_steps, sample_size=1)
    y = [0]

    for i in range(normal_sample_size - 1):
        x2, _ = multi_var_ben_ano_gen(time_steps=time_steps, sample_size=1)
        x = np.concatenate((x, x2), axis=0)
        y = np.concatenate((y, [0]), axis=0)

    if ano_sample_size > 0:
        for i in range(ano_sample_size):
            _, x2 = multi_var_ben_ano_gen(time_steps=time_steps, sample_size=1)
            x = np.concatenate((x, x2), axis=0)
            y = np.concatenate((y, [1]), axis=0)

    return x.astype(np.float32), y.astype(np.float32)


def main(_):
    from tensorflow_gan.examples.utils import np_to_tfrecords
    x, y = generate_time_series(time_steps=FLAGS.time_steps, normal_sample_size=FLAGS.normal_sample_size, ano_sample_size=FLAGS.ano_sample_size)
    for i in range(100000):
        np_to_tfrecords(x,y,"/tmp/test/" + str(i))
    #np.savetxt(FLAGS.data_file_path, x, delimiter=",")
    #np.savetxt(FLAGS.labels_file_path, y, delimiter=",")

if __name__ == '__main__':
    app.run(main)