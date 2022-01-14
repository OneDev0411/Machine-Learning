#!/usr/bin/env python3
"""Class that represents an LSTM unit"""
import numpy as np


class LSTMCell:
    """ A class that represents an LSTM unit"""

    def __init__(self, i, h, o):
        """ i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs """
        self.Wf = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wu = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wc = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wo = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wy = np.random.normal(0.0, 1.0, (h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ x_t: numpy.ndarray of shape (m, i)
                 containing the data input for the cell
                 m: batch size for the data
            h_prev: numpy.ndarray of shape (m, h)
                    containing the previous hidden state
            c_prev: numpy.ndarray of shape (m, h)
                    containing the previous cell state """
        forget_gate = self.sigmoid(
            np.matmul((np.concatenate(
                    (h_prev, x_t), axis=1)), self.Wf) + self.bf)
        update_gate = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wu) + self.bu)
        intermediate_cell_state = np.tanh(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wc) + self.bc)
        c_next = forget_gate * c_prev + update_gate * intermediate_cell_state
        output_gate = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wo) + self.bo)
        h_next = output_gate * np.tanh(c_next)
        y = self.softmax(self.by + np.matmul(h_next, self.Wy))
        return h_next, c_next, y

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
