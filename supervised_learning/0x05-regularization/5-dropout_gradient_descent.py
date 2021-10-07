#!/usr/bin/env python3
"""function that  updates the weights of a neural network with Dropout
regularization using gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """keep_prob is the probability that a node will be kept
    L is the number of layers of the network"""
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = np.matmul(dz, cache["A" + str(i - 1)].T) / Y.shape[1]
        db = np.sum(dz, axis=1, keepdims=True) / Y.shape[1]
        weights["W" + str(i)] -= alpha * dw
        weights["b" + str(i)] -= alpha * db
        dz = np.matmul(weights["W" + str(i)].T, dz) * (
                1 - cache["A" + str(i - 1)] * cache["A" + str(i - 1)])
        if i > 1:
            dz = (dz * cache["D" + str(i - 1)]) / keep_prob
