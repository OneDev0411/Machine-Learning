#!/usr/bin/env python3
""" function that calculates the
expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ X: numpy.ndarray of shape (n, d) contains the data set
        pi: numpy.ndarray of shape (k,) contains the priors for each cluster
        m: numpy.ndarray of shape (k, d) contains the centroid means"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if X.shape[1] != S.shape[1] or S.shape[
                1] != S.shape[2] or not np.isclose(np.sum(pi), 1):
        return (None, None)
    if X.shape[1] != m.shape[1] or m.shape[0] != S.shape[
                0] or pi.shape[0] != m.shape[0]:
        return (None, None)
    P = [pdf(X, m[i], S[i]) * pi[i] for i in range(pi.shape[0])]
    g = P / np.sum(P, axis=0)
    return g, np.sum(np.log(np.sum(P, axis=0)))
