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
        self.W = np.random.normal(0.0, 1.0, (1, nx))
        self.b = 0
        self.A = 0
