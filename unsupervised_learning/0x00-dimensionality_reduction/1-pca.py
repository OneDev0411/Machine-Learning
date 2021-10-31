#!/usr/bin/env python3
"""function that performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """ X: numpy.ndarray(n, d)
            n: number of data points
            d: number of dimensions in each point
        ndim: new dimensionality of the transformed X
        Returns: Returns: numpy.ndarray(n, ndim) containing
                 the transformed version of X"""
    X = X - np.mean(X, axis=0)
    vh = np.linalg.svd(X, full_matrices=True)[2]
    return np.matmul(X, vh[:ndim].T)
