#!/usr/bin/env python3
"""function that calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """ references: list of reference translations
                    each reference translation is a list of
                    the words in the translation
        sentence: list containing the model proposed sentence """
    w = len(sentence)
    reflen = len(references[np.argmin(np.abs((np.array([len(r) for r in references])) - w))])
    if w < reflen:
        brevity_p = np.exp(1 - (reflen / w))
    else:
        brevity_p = 1
    wordfreq = dict()
    for word in sentence:
        for reference in references:
            if word in wordfreq:
                if wordfreq[word] < reference.count(word):
                    wordfreq.update({word: reference.count(word)})
            else:
                wordfreq.update({word: reference.count(word)})
    return brevity_p * sum(wordfreq.values()) / w
