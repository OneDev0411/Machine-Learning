#!/usr/bin/env python3
""" function that initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """ X: array of shape (n, d) containing
            the dataset that will be used for K-means clustering
            n: number of data points
            d: number of dimensions for each data point
        k:  positive integer containing the number of clusters"""
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        return np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0), (k, X.shape[1]))
    except Exception:
        return None
