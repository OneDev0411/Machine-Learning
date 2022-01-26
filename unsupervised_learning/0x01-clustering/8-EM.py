#!/usr/bin/env python3
""" function  that performs the expectation maximization for a GMM """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ X: numpy.ndarray of shape (n, d) containing the data set
        k:  the number of clusters
        iterations:  maximum number of iterations for the algorithm
        tol: tolerance of the log likelihood, used to determine early stopping
        verbose: boolean that determines if information should be printed"""
    if not isinstance(X, np.ndarray) or len(
                X.shape) != 2 or not isinstance(k, int) or k < 1:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0 or not isinstance(
            verbose, bool) or not isinstance(
            iterations, int) or iterations < 1:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    g, lk = expectation(X, pi, m, S)
    for i in range(iterations):
        prev_l = lk
        if verbose:
            if (i % 10) == 0:
                print('Log Likelihood after {} iterations: {}'.format(
                    i, lk.round(5)))
        pi, m, S = maximization(X, g)
        g, lk = expectation(X, pi, m, S)
        if abs(lk - prev_l) <= tol:
            break
    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i + 1, lk.round(5)))
    return pi, m, S, g, lk
