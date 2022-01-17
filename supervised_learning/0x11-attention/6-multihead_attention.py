#!/usr/bin/env python3
""" multi head attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to perform multi head attention"""
    def __init__(self, dm, h):
        """ dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
            dm: divisible by h """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // self.h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """ Q: tensor of shape (batch, seq_len_q, dk)
                containing the input to generate the query matrix
            K: tensor of shape (batch, seq_len_v, dk)
                containing the input to generate the key matrix
            V: tensor of shape (batch, seq_len_v, dv)
                containing the input to generate the value matrix
            mask: always None"""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = tf.reshape(q, (batch_size, -1, self.h, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (batch_size, -1, self.h, self.depth))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, (batch_size, -1, self.h, self.depth))
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        return self.linear(
            tf.reshape(output, (batch_size, -1, self.dm))), weights
