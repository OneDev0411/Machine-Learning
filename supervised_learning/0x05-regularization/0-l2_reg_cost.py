#!/usr/bin/env python3
"""function that calculates the cost
of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """cost is the cost of the network without L2 regularization
    lambtha is the regularization parameter
    weights is a dictionary of the weights and biases
    L is the number of layers in the neural network
    m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization"""
    w = []
    for i in range(L):
        w.append(np.linalg.norm(weights['W' + str(i+1)]))
    return cost + (lambtha / (2 * m)) * np.sum(w)
