#!/usr/bin/env python3
"""function that that shuffles the data points
in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """X is the numpy.ndarray of shape (m, nx) to shuffle
    Y is the second numpy.ndarray of shape (m, ny) to shuffle"""
    p = np.random.permutation(len(X))
    return X[p], Y[p]
