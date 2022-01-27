#!/usr/bin/env python3
""" function that determines if a markov chain is absorbing """
import numpy as np


def absorbing(P):
    """ P: numpy.ndarray of shape (n, n) representing the transition matrix
        n: number of states in the markov chain """
    try:
        if not np.all(np.isclose(P.sum(axis=1), 1)):
            return False
        return np.any(P.diagonal() == 1)
    except Exception:
        return False
