#!/usr/bin/env python3
"""Class that represents a cell of a simple RNN"""
import numpy as np


class RNNCell:
    """ A class that represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """ i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs """
        self.Wh = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wy = np.random.normal(0.0, 1.0, (h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ x_t: numpy.ndarray of shape (m, i)
                 containing the data input for the cell
                 m: batche size for the data
            h_prev: numpy.ndarray of shape (m, h)
                    containing the previous hidden state """
        h_next = np.tanh(
            np.matmul((
                np.concatenate((h_prev, x_t), axis=1)), self.Wh) + self.bh)
        y = self.softmax(self.by + np.matmul(h_next, self.Wy))
        return h_next, y

    @staticmethod
    def softmax(x):
        """softmax function"""
        return np.exp(
            x - np.max(x)) / np.sum(
            np.exp(x - np.max(x)), axis=1, keepdims=True)
