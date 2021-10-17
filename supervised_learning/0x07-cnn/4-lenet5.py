#!/usr/bin/env python3
"""function  that builds a modified
version of the LeNet-5 architecture using tensorflow"""
import tensorflow as tf


def lenet5(x, y):
    """x: (m, 28, 28, 1) contains input images for the network
        m: number of images
    y: (m, 10) contains one-hot labels for the network
    returns:
        tensor for the softmax activated output
        training operation that utilizes Adam optimization
        tensor for the loss of the netowrk
         tensor for the accuracy of the network"""
    kernel = tf.contrib.layers.variance_scaling_initializer()
    L1_conv = tf.layers.conv2d(inputs=x,
                               filters=6,
                               kernel_size=(5, 5),
                               kernel_initializer=kernel,
                               padding="SAME",
                               activation='relu')
    L2_pool = tf.layers.max_pooling2d(inputs=L1_conv,
                                      pool_size=(2, 2),
                                      strides=(2, 2))
    L3_conv = tf.layers.conv2d(inputs=L2_pool,
                               filters=16,
                               kernel_size=(5, 5),
                               kernel_initializer=kernel,
                               activation='relu')
    L4_pool = tf.layers.max_pooling2d(inputs=L3_conv,
                                      pool_size=(2, 2),
                                      strides=(2, 2))
    L4_flat = tf.contrib.layers.flatten(L4_pool)
    L5_fc = tf.layers.dense(inputs=L4_flat,
                            units=120,
                            kernel_initializer=kernel,
                            activation='relu')
    L6_fc = tf.layers.dense(inputs=L5_fc,
                            units=84,
                            kernel_initializer=kernel,
                            activation='relu')
    L7_fc = tf.layers.dense(inputs=L6_fc,
                            units=10,
                            kernel_initializer=kernel,)
    Y_pred = tf.nn.softmax(L7_fc)
    loss = tf.losses.softmax_cross_entropy(y, L7_fc)
    Train_op = tf.train.AdamOptimizer().minimize(loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(L7_fc, axis=1),
                                          tf.argmax(y, axis=1)), "float32"))
    return Y_pred, Train_op, loss, acc
