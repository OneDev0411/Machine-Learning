#!/usr/bin/env python3
"""function that uses epsilon-greedy to determine the next action"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """ Q: numpy.ndarray containing the q-table
        state: the current state
        epsilon: the epsilon to use for the calculation """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(len(Q[state]))
    else:
        return np.argmax(Q[state, :])
