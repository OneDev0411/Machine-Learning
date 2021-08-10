#!/usr/bin/env python3
"""a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """nx is the number of input features to the neuron
        layers is a list representing the number of nodes in each layer"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                a = nx
            else:
                a = layers[i - 1]
            self.weights['W' + str(i + 1)
                         ] = np.random.randn(layers[i], a) * np.sqrt(2 / a)
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
