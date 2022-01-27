#!/usr/bin/env python3
""" function that determines the steady state probabilities
of a regular markov chain """
import numpy as np


def regular(P):
    """ P: numpy.ndarray of shape (n, n) representing the transition matrix
        n: number of states in the markov chain """
    try:
        dim = P.shape[0]
        q = (P-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q,ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)
    except Exception:
        return None
