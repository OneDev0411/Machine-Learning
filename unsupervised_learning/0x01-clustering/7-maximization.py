#!/usr/bin/env python3
""" function that calculates the maximization
step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """ X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) contains the posterior probability
        for each data point in each cluster"""
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2 or not isinstance(
                g, np.ndarray) or len(g.shape) != 2 or X.shape[0] != g.shape[1]:
            return None, None, None
        pi = g.sum(axis=1) / X.shape[0]
        m = np.dot(g, X) / g.sum(axis=1)[:, np.newaxis]
        S = np.zeros((m.shape[0], m.shape[1], m.shape[1]))
        for i in range(g.shape[0]):
            S[i] = np.dot(((X - m[i]) * g[i, :, np.newaxis]).T, (X - m[
                i])) / g.sum(axis=1)[i]
        return pi, m, S
    except Exception:
        return None, None, None
