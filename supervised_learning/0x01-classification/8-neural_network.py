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
        elif nx < 1:
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(0.0, 1.0, (nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(0.0, 1.0, (1, nodes))
        self.b2 = 0
        self.A2 = 0
