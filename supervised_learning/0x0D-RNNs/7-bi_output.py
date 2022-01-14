#!/usr/bin/env python3
"""Class that represents a bidirectional cell of an RNN"""
import numpy as np


class BidirectionalCell:
    """ A class that represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """ i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs """
        self.Whf = np.random.normal(0.0, 1.0, (i+h, h))
        self.Whb = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wy = np.random.normal(0.0, 1.0, (2*h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ x_t: numpy.ndarray of shape (m, i)
                 containing the data input for the cell
                 m: batch size for the data
            h_prev: numpy.ndarray of shape (m, h)
                    containing the previous hidden state """
        h_next = np.tanh(
            np.matmul((
                np.concatenate((h_prev, x_t), axis=1)), self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ x_t: numpy.ndarray of shape (m, i)
                 containing the data input for the cell
                 m: batch size for the data
            h_next: numpy.ndarray of shape (m, h)
                    containing the next hidden state """
        h_prev = np.tanh(
            np.matmul((
                np.concatenate((h_next, x_t), axis=1)), self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """ H: numpy.ndarray of shape (t, m, 2 * h) containing
            the concatenated hidden states from both directions
                t: number of time steps
                m: batch size for the data
                h: dimensionality of the hidden states """
        y = self.softmax(self.by + np.matmul(H, self.Wy))
        return y

    @staticmethod
    def softmax(x):
        """softmax function"""
        return np.exp(
            x - np.max(x)) / np.sum(
            np.exp(x - np.max(x)), axis=2, keepdims=True)
