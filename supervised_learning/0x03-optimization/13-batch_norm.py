#!/usr/bin/env python3
"""batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """function that normalizes an unactivated
    output of a neural network using batch normalization"""
    m = np.mean(Z, axis=0)
    v = np.var((Z - m), axis=0)
    normalized = (Z - m) / (np.sqrt(v + epsilon))
    return gamma * normalized + beta
