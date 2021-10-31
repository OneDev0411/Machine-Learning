#!/usr/bin/env python3
"""function that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """ X: numpy.ndarray of shape (n, d)
            n: number of data points
            d: number of dimensions in each point
        var: fraction of the variance that the PCA transformation
             should maintain """
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    variance = np.cumsum(s) / np.sum(s)
    s = np.sum(np.where(variance <= var, 1, 0))
    return vh[:s+1].T
