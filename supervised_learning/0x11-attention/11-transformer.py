#!/usr/bin/env python3
""" Transformer """
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """ Class that inherits from tensorflow.keras.layers.Layer
        to create the encoder for a transformer"""

    def __init__(
            self,
            N,
            dm,
            h,
            hidden,
            input_vocab,
            target_vocab,
            max_seq_input,
            max_seq_target,
            drop_rate=0.1):
        """ N - the number of blocks in the encoder and decoder
            dm - the dimensionality of the model
            h - the number of heads
            hidden - the number of hidden units in the fully connected layers
            input_vocab - the size of the input vocabulary
            target_vocab - the size of the target vocabulary
            max_seq_input - maximum sequence length possible for the input
            max_seq_target - maximum sequence length possible for the target
            drop_rate - the dropout rate"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate)
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(
            self,
            inputs,
            target,
            training,
            encoder_mask,
            look_ahead_mask,
            decoder_mask):
        """ inputs - a tensor of shape containing the inputs
            target - a tensor containing the target
            training - a boolean to determine if the model is training
            encoder_mask - the padding mask to be applied to the encoder
            look_ahead_mask - the look ahead mask to be applied to the decoder
            decoder_mask - the padding mask to be applied to the decoder"""
        encoded = self.encoder(inputs, training, encoder_mask)
        decoded = self.decoder(
            target,
            encoded,
            training,
            look_ahead_mask,
            decoder_mask)
        transform = self.linear(decoded)
        return transform
