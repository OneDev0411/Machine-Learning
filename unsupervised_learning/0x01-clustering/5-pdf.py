#!/usr/bin/env python3
""" function that calculates the probability density function of a
Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """ X: numpy.ndarray of shape (n, d) containing the data points
        whose PDF should be evaluated
        m: numpy.ndarray of shape (d,) containing the mean of the
        distribution
        S: numpy.ndarray of shape (d, d) containing the covariance
        of the distribution"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    d = X.shape[1]
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    x = np.exp(- np.dot(np.dot((X - m), np.linalg.inv(S)), (
                X - m).T).diagonal() / 2)
    y = np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
    return np.where((x / y) < 1e-300, 1e-300, (x / y))
