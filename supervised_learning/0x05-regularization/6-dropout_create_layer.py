#!/usr/bin/env python3
"""function that creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function
    keep_prob is the probability that a node will be kept
    Returns: the output of the new layer"""
    layer = tf.layers.Dense(
        n,
        activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        kernel_regularizer=tf.layers.Dropout(keep_prob))
    return layer(prev)
