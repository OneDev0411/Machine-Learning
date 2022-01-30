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
        mu, cov = self.gp.predict(self.X_s)
        z = np.zeros(cov.shape[0])
        if self.minimize:
            mu_ = np.min(self.gp.Y)
            ip = mu_ - mu - self.xsi
        else:
            mu_s_opt = np.max(self.gp.Y)
            ip = mu - mu_s_opt - self.xsi
        for i in range(cov.shape[0]):
            if cov[i] > 0:
                z[i] = ip[i] / cov[i]
            else:
                z[i] = 0
            EI = ip * norm.cdf(z) + cov * norm.pdf(z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

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
        self.gp.X = self.gp.X[:-1]
        return self.gp.X[opt], self.gp.Y[opt]
