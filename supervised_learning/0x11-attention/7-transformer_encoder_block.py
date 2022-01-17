#!/usr/bin/env python3
""" multi head attention """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to perform multi head attention"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
            dm: divisible by h """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """ x: tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder block
            training: boolean to determine if the model is training
            mask: mask to be applied for multi head attention"""
        output, weights = self.mha(x, x, x, mask)
        output = self.dropout1(output, training=training)
        output_n1 = self.layernorm1(x+output)
        output = self.dense_hidden(output_n1)
        output = self.dense_output(output)
        output = self.dropout2(output, training=training)
        output_n2 = self.layernorm2(output_n1 + output)
        return output_n2
