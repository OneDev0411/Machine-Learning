#!/usr/bin/env python3
"""Class that represents a gated recurrent unit"""
import numpy as np


class GRUCell:
    """ A class that represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """ i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs """
        self.Wz = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wr = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wh = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wy = np.random.normal(0.0, 1.0, (h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ x_t: numpy.ndarray of shape (m, i)
                 containing the data input for the cell
                 m: batche size for the data
            h_prev: numpy.ndarray of shape (m, h)
                    containing the previous hidden state """
        update_gate = self.sigmoid(
            np.matmul((np.concatenate(
                    (h_prev, x_t), axis=1)), self.Wz) + self.bz)
        reset_gate = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wr) + self.br)
        cmc = np.tanh(
            np.matmul((np.concatenate(
                ((reset_gate * h_prev), x_t), axis=1)), self.Wh) + self.bh)
        h_next = h_prev * (1 - update_gate) + update_gate * cmc
        y = self.softmax(self.by + np.matmul(h_next, self.Wy))
        return h_next, y

    @staticmethod
    def softmax(x):
        """softmax function"""
        return np.exp(
            x - np.max(x)) / np.sum(
            np.exp(x - np.max(x)), axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))
