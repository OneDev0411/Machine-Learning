#!/usr/bin/env python3
"""function that creates a layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use"""
    layer = tf.layers.Dense(
        n,
        activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="Layer")
    return layer(prev)
