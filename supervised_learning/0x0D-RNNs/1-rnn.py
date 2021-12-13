#!/usr/bin/env python3
"""Class that represents a cell of a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ rnn_cell: instance of RNNCell that will
                be used for the forward propagation
        X: data to be used, given as a
            numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: the batch size
            i: dimensionality of the data
        h_0: initial hidden state, given as a
            numpy.ndarray of shape (m, h)
            h: dimensionality of the hidden state """
    t, m, i = X.shape
    m, h = h_0.shape
    o = rnn_cell.Wy.shape[1]
    H = np.zeros(shape=(t + 1, m, h))
    Y = np.zeros(shape=(t, m, o))
    H[0] = h_0
    for i in range(t):
        H[i+1, :, :], Y[i, :, :] = rnn_cell.forward(H[i, :, :], X[i, :, :])
    return H, Y
