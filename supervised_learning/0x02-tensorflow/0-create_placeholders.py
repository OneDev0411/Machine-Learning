#!/usr/bin/python3
"""function that returns two placeholders,
 x and y, for the neural network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    x = tf.placeholder("float", [None, nx], "x")
    y = tf.placeholder("float", [None, classes], "y")
    return x, y
