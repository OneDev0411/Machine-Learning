#!/usr/bin/env python3
""" function that determines if a markov chain is absorbing """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Observation: numpy.ndarray of shape (T,)
        Emission: numpy.ndarray of shape (N, M)
        Transition: numpy.ndarray of shape (N, N)
        Initial: numpy.ndarray of shape (N, 1)"""
    T = Observation.shape[0]
    N = Emission.shape[0]
    viterbi = np.zeros((N, T))
    vit = np.zeros((N, T))
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            prob = viterbi[:, i-1] * Emission[
                    j, Observation[i]] * Transition[:, j]
            viterbi[j, i] = np.max(prob)
            vit[j, i] = np.argmax(prob, 0)
    path = [np.argmax(viterbi[:, T - 1])] + []
    prev = np.argmax(viterbi[:, T - 1])
    for i in range(T - 1, 0, -1):
        prev = int(vit[prev, i])
        path = [prev] + path
    return path, np.max(viterbi[:, T - 1])
