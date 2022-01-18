#!/usr/bin/env python3
""" Decoder """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to create the encoder for a transformer"""

    def __init__(
            self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """ N - the number of blocks in the encoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in
                    the fully connected layer
            target_vocab - the size of the target vocabulary
            max_seq_len - the maximum sequence length possible
            drop_rate - the dropout rate"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for i in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """  x: tensor of shape (batch, target_seq_len, dm)
                containing the input to the decoder
            encoder_output: tensor of shape (batch, input_seq_len, dm)
                            containing the output of the encoder
            training: boolean to determine if the model is training
            look_ahead_mask: mask to be applied to
                                the first multi head attention layer
            padding_mask: mask to be applied to the
                        second multi head attention layer"""
        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = x + self.positional_encoding[:x.shape[1], :]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask)
        return x
