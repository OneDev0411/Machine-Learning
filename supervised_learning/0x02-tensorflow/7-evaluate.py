#!/usr/bin/env python3
"""function that that evaluates the output of a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path)
        saver.restore(sess, save_path)
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        return y_pred, loss, accuracy
