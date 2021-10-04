#!/usr/bin/env python3
"""batch normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """function that that creates a batch
    normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, name="layer",
                            kernel_initializer=kernel)
    mean, variance = tf.nn.moments(layer(prev), [0])
    gamma = tf.ones([n])
    beta = tf.zeros([n])
    norm = tf.nn.batch_normalization(
        layer(prev), mean, variance, beta, gamma, 1e-8)
    return activation(norm)
