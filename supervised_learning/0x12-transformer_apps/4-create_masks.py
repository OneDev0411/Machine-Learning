#!/usr/bin/env python3
"""Function that creates all masks for training/validation"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """ inputs: Tensor of shape (batch_size, seq_len_in)
                contains the input sentence
        target: Tensor of shape (batch_size, seq_len_out)
                contains the target sentence """
    batch_size, seq_len_out = target.shape
    batch_size, seq_len_in = inputs.shape
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones(shape=(
            batch_size, 1, seq_len_out, seq_len_out)), -1, 0)
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[
                              :, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return encoder_mask, combined_mask, decoder_mask
