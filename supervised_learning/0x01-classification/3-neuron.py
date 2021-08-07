#!/usr/bin/env python3
"""neuron performing binary classification"""
import numpy as np


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
        """Calculates the forward propagation of the neuron
        X: numpy.ndarray with shape (nx, m) that contains the input data"""
        self.__A = np.dot(self.__W, X) + self.__b
        self.__A = self.sig(self.__A)
        return self.__A

    @staticmethod
    def sig(x):
        """sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray containing the correct labels for the input data
        A is a numpy.ndarray containing the activated output of the neuron for each example"""
        (a, m) = np.shape(Y)
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y)
                               * (np.log(1.0000001 - A)))
        return cost
