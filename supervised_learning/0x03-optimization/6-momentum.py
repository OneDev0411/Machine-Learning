#!/usr/bin/env python3
"""momentum optimization function"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Function that creates the training operation for
    a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
