#!/usr/bin/env python3
""" A function that calculates the scaled dot product attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ Q: tensor with its last two dimensions as (..., seq_len_q, dk)
            containing the query matrix
        K: tensor with its last two dimensions as (..., seq_len_v, dk)
            containing the key matrix
        V: tensor with its last two dimensions as (..., seq_len_v, dv)
            containing the value matrix
        mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing the optional mask, or defaulted to None"""
    smm = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
        tf.cast(tf.shape(K)[-1], tf.float32))
    if mask:
        smm += (mask * -1e9)
    W = tf.nn.softmax(smm, axis=-1)
    return tf.matmul(W, V), W
