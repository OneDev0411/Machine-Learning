#!/usr/bin/env python3
"""Function that calculates the mean and covariance of a data set"""
import numpy as np

def mean_cov(X):
    """ X: numpy.ndarray of shape (n, d)
    n: number of data points
    d: number of dimensions in each data point"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = X.mean(axis=0)
    cov = np.matmul((X-mean).T, X-mean) / (n-1)
    return mean, cov
