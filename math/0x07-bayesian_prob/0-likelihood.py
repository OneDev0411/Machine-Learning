#!/usr/bin/env python3
""" function that calculates the likelihood of obtaining
 this data given various hypothetical probabilities
 of developing severe side effects"""
import numpy as np


def factorial(x):
    """ function that calculate the factorial of x """
    fact = 1
    for i in range(1, x + 1):
        fact = fact * i
    return fact


def likelihood(x, n, P):
    """ x: number of patients that develop severe side effects
        n: the total number of patients observed
        P: 1D numpy.ndarray containing the various hypothetical
            probabilities of developing severe side effects """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         "greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if (P > 1).any() or (P < 0).any():
        raise ValueError("All values in P must be in "
                         "the range [0, 1]")
    return (factorial(n) / (
            factorial(x) * factorial(n - x))) * (
            P ** x) * ((1 - P) ** (n - x))
