#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """function that that creates the training operation
    for a neural network in tensorflow using
    the RMSProp optimization algorithm"""
    return tf.train.RMSPropOptimizer(alpha, beta2, epsilon).minimize(loss)
