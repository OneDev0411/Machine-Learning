#!/usr/bin/env python3
""" function that determines the steady state probabilities
of a regular markov chain """
import numpy as np


def regular(P):
    """ P: numpy.ndarray of shape (n, n) representing the transition matrix
        n: number of states in the markov chain """
    try:
        if not np.all(np.isclose(P.sum(axis=1), 1)):
            return None
        dim = P.shape[0]
        steady = np.zeros((1, dim))
        q = (P-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q,ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        solve = np.linalg.solve(QTQ,bQT)
        for i in range(dim):
            steady[0][i] = solve[i]
        return steady
    except Exception:
        return None
