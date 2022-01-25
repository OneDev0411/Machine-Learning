#!/usr/bin/env python3
""" function that initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters"""
    try:
        if len(X.shape) != 2:
            return None, None, None
        S = np.array([np.identity(X.shape[1])] * k)
        return np.full(k, 1 / k), kmeans(X, k)[0], S
    except Exception:
        return None, None, None
