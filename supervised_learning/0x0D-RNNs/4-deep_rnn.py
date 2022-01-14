#!/usr/bin/env python3
"""function that performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ rnn_cells: list of RNNCell instances of length l.
        X: the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (l, m, h)
            h: dimensionality of the hidden state
        Returns: H, Y"""
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t+1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, rnn_cells[l-1].by.shape[1]))
    for i in range(t):
        x_prev = X[i]
        for j in range(l):
            x_t = x_prev
            h_prev = H[i, j]
            x_prev, y = rnn_cells[j].forward(h_prev, x_t)
            H[i+1, j, :, :] = x_prev
        Y[i] = y
    return H, Y
