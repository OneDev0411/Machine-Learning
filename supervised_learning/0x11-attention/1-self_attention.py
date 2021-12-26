#!/usr/bin/env python3
""" calculates the attention for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to calculate the attention for machine translation"""
    def __init__(self, units):
        """ units: integer representing the number of hidden units
            in the alignment model"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """ s_prev: tensor of shape (batch, units)
                    containing the previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
                            containing the outputs of the encoder """
        s_prev = tf.expand_dims(s_prev, 1)
        V = self.V(tf.nn.tanh(self.W(s_prev) + self.U(
            hidden_states)))
        W = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(W * hidden_states, axis=1)
        return context, W
