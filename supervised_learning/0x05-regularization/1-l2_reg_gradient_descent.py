#!/usr/bin/env python3
"""function that updates the weights and biases of
 a neural network using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """y is a one-hot array of shape (classes, m) containing the
    correct labels
    weights is a dictionary of the weights and biases
    cache is a dictionary of the outputs of each layer
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    The neural network uses tanh activations
     on each layer except the last, which uses a softmax activation
    The weights and biases of the network should be updated in place"""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dz = np.matmul(weights["W" + str(i)].T, dz) * (
                1 - cache['A' + str(i - 1)] * cache['A' + str(i - 1)])
        weights["W" + str(i)] -= alpha * \
            (dw + (lambtha / m) * weights["W" + str(i)])
        weights["b" + str(i)] -= alpha * db
