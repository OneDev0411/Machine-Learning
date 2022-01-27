#!/usr/bin/env python3
""" function that determines the probability of a markov chain being
in a particular state after a specified number of iterations """
import numpy as np


def markov_chain(P, s, t=1):
    """ P: numpy.ndarray of shape (n, n) representing the transition matrix
        n: number of states in the markov chain
        s: numpy.ndarray of shape (1, n) representing the
        probability of starting in each state
        t: number of iterations that the markov chain has been through"""
    for i in range(t):
        s = np.matmul(s, P)
    return s
