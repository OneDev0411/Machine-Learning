#!/usr/bin/env python3
""" function that performs K-means on a dataset"""
import numpy as np


def initialize(X, k):
    """ Function that initializes cluster centroids for K-means """
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        return np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0), (k, X.shape[1]))
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """ X: array of shape (n, d) containing
            the dataset that will be used for K-means clustering
            n: number of data points
            d: number of dimensions for each data point
        k:  positive integer containing the number of clusters
        iterations: maximum number of iterations"""
    C = initialize(X, k)
    if C is None or not isinstance(iterations, int) or iterations <= 0:
        return None, None
    clss = np.linalg.norm((X[:, np.newaxis, :] - C), axis=2).argmin(axis=1)
    for i in range(iterations):
        Ccopy = np.copy(C)
        for j in range(k):
            if X[clss == j].size == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = X[clss == j].mean(axis=0)
        clss = np.linalg.norm((X[:, np.newaxis, :] - C), axis=2).argmin(axis=1)
        if (Ccopy == C).all():
            return C, clss
    return C, clss
