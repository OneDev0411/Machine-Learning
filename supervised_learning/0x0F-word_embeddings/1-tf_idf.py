#!/usr/bin/env python3
""" A function that creates a bag of words embedding matrix """
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis"""
    TfidfVec = TfidfVectorizer(ngram_range=(1, 1), vocabulary=vocab)
    embeddings = TfidfVec.fit_transform(sentences).toarray()
    features = TfidfVec.get_feature_names()
    return embeddings, features
