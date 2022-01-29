#!/usr/bin/env python3
"""Class that represents a noiseless 1D Gaussian process"""
import numpy as np


class GaussianProcess:
    """ A class that represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
            Y_init:  numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
            t: number of initial samples
            l:  length parameter for the kernel
            sigma_f: standard deviation given to the output of
                    the black-box function """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ X1:  numpy.ndarray of shape (m, 1)
            X2:  numpy.ndarray of shape (n, 1) """
        sq = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(
            X1, X2.T)
        return (self.sigma_f ** 2) * np.exp((-1 / (2 * self.l ** 2)) * sq)

    def predict(self, X_s):
        """X_s: numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated"""
        kernel = self.kernel(self.X, X_s)
        s_kernel = self.kernel(X_s, X_s)
        kernel_inv = np.linalg.inv(self.K)
        return kernel.T.dot(kernel_inv).dot(self.Y).reshape(-1), (
            s_kernel - kernel.T.dot(kernel_inv).dot(kernel)).diagonal()

    def update(self, X_new, Y_new):
        """X_new: numpy.ndarray of shape (1,) representing the new sample point
        Y_new: numpy.ndarray of shape (1,) representing the new
                sample function value"""
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
