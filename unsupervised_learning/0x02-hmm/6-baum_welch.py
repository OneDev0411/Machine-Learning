#!/usr/bin/env python3
""" function that performs the Baum-Welch
algorithm for a hidden markov model"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ Observation: numpy.ndarray of shape (T,)
        Emission: numpy.ndarray of shape (N, M)
        Transition: numpy.ndarray of shape (N, N)
        Initial: numpy.ndarray of shape (N, 1)"""
    pass
