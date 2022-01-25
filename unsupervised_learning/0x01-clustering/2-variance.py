#!/usr/bin/env python3
""" function that calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """ X: ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) contains centroid means of clusters"""
    try:
        clss = np.linalg.norm((X[:, np.newaxis, :] - C), axis=2).argmin(axis=1)
        var = np.square(np.linalg.norm(X - C[clss]))
        return var
    except Exception:
        return None
