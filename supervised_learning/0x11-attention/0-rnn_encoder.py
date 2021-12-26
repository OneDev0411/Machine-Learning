#!/usr/bin/env python3
""" RNN encoder """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to encode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """ vocab: integer representing the size of the input vocabulary
            embedding: integer representing the dimensionality of
                        the embedding vector
            units: integer representing the number of hidden units
                    in the RNN cell
            batch: integer representing the batch size """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """ Initializes the hidden states for the RNN cell
            to a tensor of zeros """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """ Returns outputs and the last hidden state of the encoder
            x: tensor of shape (batch, input_seq_len)
                containing the input to the encoder layer
            initial: tensor of shape (batch, units)
                    containing the initial hidden state """
        x = self.embedding(x)
        return self.gru(x, initial)
