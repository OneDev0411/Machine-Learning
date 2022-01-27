#!/usr/bin/env python3
""" function that determines if a markov chain is absorbing """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ Observation: numpy.ndarray of shape (T,)
        Emission: numpy.ndarray of shape (N, M)
        Transition: numpy.ndarray of shape (N, N)
        Initial: numpy.ndarray of shape (N, 1)"""
    T = Observation.shape[0]
    F = np.zeros((Emission.shape[0], T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        F[:, i] = Emission[:, Observation[i]] * np.dot(Transition.T, F[:, i-1])
    return F[:, T-1].sum(), F
