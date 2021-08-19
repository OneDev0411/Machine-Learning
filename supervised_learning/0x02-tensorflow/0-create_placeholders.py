#!/usr/bin/env python3
"""function that returns two placeholders,
 x and y, for the neural network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """x is the placeholder for the input data to the neural network,
    y is the placeholder for the one-hot labels for the input data"""
    x = tf.placeholder("float", [None, nx], "x")
    y = tf.placeholder("float", [None, classes], "y")
    return x, y
