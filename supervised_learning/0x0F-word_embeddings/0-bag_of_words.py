#!/usr/bin/env python3
""" A function that creates a bag of words embedding matrix """
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words(sentences, vocab=None):
    """ sentences: list of sentences to analyze
        vocab: list of the vocabulary words to use for the analysis"""
    """s = len(sentences)
    for sentence in sentences:
        sentence = sentence.lower()
        words = sentence.split()
        words = [word.strip('.,!;()[]') for word in words]
        words = [word.replace("'s", '') for word in words]"""

    CountVec = CountVectorizer(ngram_range=(1, 1), vocabulary=vocab)
    embeddings = CountVec.fit_transform(sentences).toarray()
    features = CountVec.get_feature_names()
    return embeddings, features
