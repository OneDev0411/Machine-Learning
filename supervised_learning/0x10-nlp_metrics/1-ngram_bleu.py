#!/usr/bin/env python3
"""function that calculates the n-gram BLEU score for a sentence"""
import numpy as np


def ngram_generator(sentence, n=2):
    """N-gram generator"""
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngrams.append(' '.join(sentence[i:i + n]))
    return ngrams


def ngram_bleu(references, sentence, n):
    """ references: list of reference translations
                    each reference translation is a list of
                    the words in the translation
        sentence: list containing the model proposed sentence
         n: size of the n-gram to use for evaluation"""
    ngram = ngram_generator(sentence, n)
    ngram_ref = list(ngram_generator(sen, n) for sen in references)
    w = len(sentence)
    reflen = len(references[np.argmin(np.abs(
        (np.array([len(r) for r in references])) - w))])
    if w < reflen:
        brevity_p = np.exp(1 - (reflen / w))
    else:
        brevity_p = 1
    wordfreq = dict()
    for word in ngram:
        for reference in ngram_ref:
            if word in wordfreq:
                if wordfreq[word] < reference.count(word):
                    wordfreq.update({word: reference.count(word)})
            else:
                wordfreq.update({word: reference.count(word)})
    return brevity_p * sum(wordfreq.values()) / len(ngram)
