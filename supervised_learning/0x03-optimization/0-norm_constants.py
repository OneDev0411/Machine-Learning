#!/usr/bin/env python3
"""function that calculates the normalization constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """X is the numpy.ndarray of shape (m, nx) to normalize
    m is the number of data points
    nx is the number of features"""
    m, nx = np.shape(X)
    mean = np.zeros(nx,)
    stdev = np.zeros(nx,)
    for i in range(nx):
        mean[i] = X[:, i].mean()
        stdev[i] = X[:, i].std()
    return mean, stdev
