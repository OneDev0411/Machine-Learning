#!/usr/bin/env python3
"""function that c that creates the training
operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """loss is the loss of the network’s prediction
    alpha is the learning rate
    Returns: an operation that trains the network
    using gradient descent"""
    grad_var = tf.train.GradientDescentOptimizer(alpha).compute_gradients(loss)
    return tf.train.GradientDescentOptimizer(alpha).apply_gradients(grad_var)
