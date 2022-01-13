#!/usr/bin/env python3
""" a script to preprocess coinbase datasets """
import pandas as pd
import numpy as np
import tensorflow as tf


df = pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')

""" data preparation """

""" removing missing data points """
df.dropna(inplace=True)
""" removing extra features and keeping only the closing price """
df.drop(['Open', 'High', 'Low',
         'Volume_(BTC)', 'Volume_(Currency)',
         'Weighted_Price'], axis=1, inplace=True)
""" converting timestamp to datetime """
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
""" Renaming columns """
df.columns = ['Time', 'Closing Price']
""" keeping only relevant years """
df['Time'] = df[df['Time'] > '2018']
df.dropna(inplace=True)
"""  changing the time window from 60s to 1hr """
df = df.set_index('Time').asfreq('1H')

""" data splitting """
n = len(df)
training_set = df[0:int(n * 0.7)]
validation_set = df[int(n * 0.7):int(n * 0.9)]
testing_set = df[int(n * 0.9):]

""" Data Normalization """


def normalize(data):
    """ Function that normalizes a dataset
     based on mean normalization """
    return (data - data.mean()) / data.std()


training_set = normalize(training_set)
validation_set = normalize(validation_set)
testing_set = normalize(testing_set)
training_set.dropna(inplace=True)
testing_set.dropna(inplace=True)
validation_set.dropna(inplace=True)

""" Data windowing / splitting and converting to tf.data.dataset """


class WindowGenerator:
    """" Class that handles the indexes and offsets
         Split windows of features into (features, labels) pairs """
    def __init__(self, input_width, label_width, shift,
                 train_df=training_set,
                 val_df=validation_set,
                 test_df=testing_set,
                 label_columns=None):
        """ Includes all the necessary logic for
            the input and label indices."""
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """ Converts features to a window of inputs and a window of labels.
            features:  list of consecutive inputs."""
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        """ function that takes a time series DataFrame and convert it to
        a tf.data.Dataset of (input_window, label_window) pairs"""
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        """ Returns the tf.data.Dataset of the training set """
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """ Returns the tf.data.Dataset of the validation set """
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """ Returns the tf.data.Dataset of the testing set """
        return self.make_dataset(self.test_df)


window = WindowGenerator(input_width=24,
                         label_width=1,
                         shift=1,
                         label_columns=['Closing Price'])
