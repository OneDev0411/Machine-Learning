#!/usr/bin/env python3
"""function that that conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(L):
        z = np.matmul(weights['W' + str(i + 1)],
                      cache['A' + str(i)]) + weights['b' + str(i + 1)]
        if i == L - 1:
            x = np.exp(z)
            cache['A' + str(i + 1)] = x / np.sum(x, axis=0)
        else:
            cache['A' + str(i + 1)] = np.tanh(z)
            cache['D' + str(i + 1)] = np.random.binomial(
                1, keep_prob, size=z.shape)
            cache['A' + str(i + 1)] *= cache['D' + str(i + 1)]
            cache['A' + str(i + 1)] /= keep_prob
    return cache
