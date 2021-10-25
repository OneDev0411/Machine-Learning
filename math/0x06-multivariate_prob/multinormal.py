#!/usr/bin/env python3
"""Function that calculates a correlation matrix"""
import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution"""
    def __init__(self, data):
        """ data: numpy.ndarray of shape (d, n)
        n: number of data points
        d: number of dimensions"""
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = data.mean(axis=1, keepdims=True)
        self.cov = np.matmul(data-self.mean, (data-self.mean).T) / (n-1)
