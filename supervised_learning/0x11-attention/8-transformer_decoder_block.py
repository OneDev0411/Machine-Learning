#!/usr/bin/env python3
""" multi head attention """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to create an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ dm: integer representing the dimensionality of the model
            h: integer representing the number of heads
            dm: divisible by h """
        super(decoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            units=hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ x: tensor of shape (batch, input_seq_len, dm)
                containing the input to the encoder block
            training: boolean to determine
                        if the model is training
            look_ahead_mask: mask to be applied to
                            the first multi head attention layer
            padding_mask: mask to be applied to
                            the second multi head attention layer"""
        output1, weights = self.mha1(x, x, x, look_ahead_mask)
        output1 = self.dropout1(output1, training=training)
        output_n1 = self.layernorm1(x+output1)
        output2, weights2 = self.mha2(
            output_n1, encoder_output, encoder_output, padding_mask)
        output2 = self.dropout2(output2, training=training)
        output_n2 = self.layernorm2(output2 + output_n1)
        output = self.dense_hidden(output_n2)
        output = self.dense_output(output)
        output = self.dropout3(output, training=training)
        output_n3 = self.layernorm3(output+output_n2)
        return output_n3
