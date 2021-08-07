#!/usr/bin/env python3

import numpy as np
"""neuron performing binary classification"""


class Neuron:
    """a single neuron performing binary classification"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0.0, 1.0, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """X: numpy.ndarray with shape (nx, m) that contains the input data"""
        self.__A = np.dot(self.__W, X) + self.__b
        self.__A = self.sig(self.__A)
        return self.__A

    @staticmethod
    def sig(x):
        """sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))
