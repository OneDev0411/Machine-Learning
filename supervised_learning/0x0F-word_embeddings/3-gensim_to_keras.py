#!/usr/bin/env python3
""" A function that converts a gensim word2vec
model to a keras Embedding layer """
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """ model: a trained gensim word2vec models
        Returns: the trainable keras Embedding"""
    model.wv.get_keras_embedding(train_embeddings=False)
