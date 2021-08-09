#!/usr/bin/env python3
"""a neural network with one hidden layer performing binary classification"""
import numpy as np


class NeuralNetwork:
    """a neural network performing binary classification"""

    def __init__(self, nx, nodes):
        """nx is the number of input features to the neuron
        nodes is the number of nodes found in the hidden layer"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0.0, 1.0, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0.0, 1.0, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X: numpy.ndarray with shape (nx, m) that contains the input data"""
        self.__A1 = self.sig(np.dot(self.__W1, X) + self.__b1)
        self.__A2 = self.sig(np.dot(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    @staticmethod
    def sig(x):
        """sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray containing the correct labels for the input data
        A is a numpy.ndarray containing the activated output"""
        (a, m) = np.shape(Y)
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) *
                               (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions
        X is a numpy.ndarray tcontaining the input data
        Y is a numpy.ndarray containing the correct labels"""
        pred1, pred2 = self.forward_prop(X)
        predic2 = np.where(pred2 >= 0.5, 1, 0)
        return predic2, self.cost(Y, pred2)

