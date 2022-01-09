#!/usr/bin/env python3
""" Class that loads and preps a dataset for machine translation """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class that loads and preps a dataset for machine translation """

    def __init__(self):
        """initialization"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)
        self.data_train = self.data_train.map(self.tf_encode).cache()
        self.data_valid = self.data_valid.map(self.tf_encode).cache()

    def tokenize_dataset(self, data):
        """ Instance method that creates sub-word tokenizers for our dataset
            data: tf.data.Dataset """
        ste = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        tokenizer_pt = ste([pt.numpy() for pt, _ in data],
                           target_vocab_size=2 ** 15)
        tokenizer_en = ste([en.numpy() for _, en in data],
                           target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ Encode a translation into tokens """
        pt_size = self.tokenizer_pt.vocab_size
        en_size = self.tokenizer_en.vocab_size
        pt_tokens = [pt_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')) + [pt_size + 1]
        en_tokens = [en_size] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')) + [en_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """ Tensorflow wrapper for the "encode" instance method """
        tfpt, tfen = tf.py_function(
            self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        tfpt.set_shape([None])
        tfen.set_shape([None])
        return tfpt, tfen
