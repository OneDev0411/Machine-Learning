#!/usr/bin/env python3
"""Class that performs Bayesian optimization on
a noiseless 1D Gaussian process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ A class that performs Bayesian optimization
    on a noiseless 1D Gaussian process"""

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """ f is the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1)
            Y_init: numpy.ndarray of shape (t, 1)
            t: number of initial samples
            bounds : tuple of (min, max)
            ac_samples: number of samples that should be analyzed
            l: length parameter for the kernel
            sigma_f: standard deviation given to the output
            xsi: exploration-exploitation factor for acquisition
            minimize: a bool determining whether optimization should be
            performed for minimization (True) or maximization (False)"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ instancethat calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            opt = np.min(self.gp.Y)
        else:
            opt = np.max(self.gp.Y)
        sig = opt - mu - self.xsi
        EI = sig * norm.cdf(sig / sigma) + sigma * norm.pdf(sig / sigma)
        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """ iterations: the maximum number of iterations to perform """
        X = []
        for i in range(iterations):
            X_next = self.acquisition()[0]
            if X_next in X:
                break
            Y_new = self.f(X_next)
            self.gp.update(X_next, Y_new)
            X.append(X_next)
        if (self.minimize):
            opt = np.argmin(self.gp.Y)
        else:
            opt = np.argmax(self.gp.Y)
        return self.gp.X[opt], self.gp.Y[opt]
