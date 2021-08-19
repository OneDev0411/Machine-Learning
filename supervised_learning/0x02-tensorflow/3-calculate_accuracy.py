#!/usr/bin/env python3
"""function that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """y is a placeholder for the labels of the input data
    y_pred is a tensor containing the network’s predictions"""
    accuracy = tf.math.reduce_mean(
        tf.cast(tf.equal(
            tf.argmax(y, axis=-1),
            tf.argmax(y_pred, axis=-1)), tf.float32))
    return accuracy
