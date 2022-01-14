#!/usr/bin/env python3
"""function that performs forward propagation for a bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ bi_cells: s an instance of BidirectinalCell that
        will be used for the forward propagation
        X: the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state in the forward direction,
        given as a numpy.ndarray of shape (m, h)
            h: dimensionality of the hidden state
        h_t: initial hidden state in the backward direction
        Returns: H, Y"""
    t, m, i = X.shape
    forward = []
    backward = []
    h_prev = h_0
    h_next = h_t
    for i in range(t):
        forward.append(bi_cell.forward(h_prev, X[i]))
        h_prev = forward[i]
        backward.append(bi_cell.backward(h_next, X[t-i-1]))
        h_next = backward[i]
    backward.reverse()
    H = np.concatenate((np.array(forward), np.array(backward)), axis=2)
    return H, bi_cell.output(H)
