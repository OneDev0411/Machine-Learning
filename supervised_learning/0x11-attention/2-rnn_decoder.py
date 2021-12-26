#!/usr/bin/env python3
""" Decode for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to decode for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """ vocab: integer representing the size of the input vocabulary
            embedding: integer representing the dimensionality of
                        the embedding vector
            units: integer representing the number of hidden units
                    in the RNN cell
            batch: integer representing the batch size """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """ x: tensor of shape (batch, 1) containing  the previous
            word in the target sequence as an index of the
            target vocabulary
            s_prev: tensor of shape (batch, units) containing the
            previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder"""
        attention = SelfAttention(self.units)
        context, W = attention(s_prev, hidden_states)
        x = self.embedding(x)
        concat = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        y, s = self.gru(concat)
        return self.F(tf.reshape(y, (-1, y.shape[2]))), s
