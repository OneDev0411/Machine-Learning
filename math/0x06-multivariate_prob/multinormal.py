#!/usr/bin/env python3
"""Function that calculates a correlation matrix"""
import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """data: numpy.ndarray of shape (d, n)
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
        self.cov = np.matmul(data - self.mean, (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """public instance method  that calculates the PDF at a data point
        x: numpy.ndarray of shape (d, 1)
        d: number of dimensions of the Multinomial instance"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise TypeError("x must have the shape ({d}, 1)")
        if x.shape[0] != self.cov.shape[0]:
            raise TypeError("x must have the shape ({d}, 1)")
        d = x.shape[0]
        den = np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        inv = np.linalg.inv(self.cov)
        ex = (-0.5 * np.matmul(
            np.matmul((x - self.mean).T, inv), x - self.mean))
        return (1 / den) * np.exp(ex[0][0])
