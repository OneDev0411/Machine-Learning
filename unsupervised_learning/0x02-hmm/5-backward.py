#!/usr/bin/env python3
""" function that performs the backward algorithm for a hidden markov mode """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ Observation: numpy.ndarray of shape (T,)
        Emission: numpy.ndarray of shape (N, M)
        Transition: numpy.ndarray of shape (N, N)
        Initial: numpy.ndarray of shape (N, 1)"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    B = np.zeros((N, T))
    B[:, T-1] = np.ones(N)
    for t in range(T-2, -1, -1):
        for j in range(N):
            B[j, t] = (B[:, t+1] * Emission[:, Observation[t+1]]).dot(
                    Transition[j, :])
    return np.sum(Initial.T * B[:, 0] * Emission[:, Observation[0]]), B
