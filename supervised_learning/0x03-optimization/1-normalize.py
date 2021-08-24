#!/usr/bin/env python3
"""function that that normalizes (standardizes) a matrix"""
import numpy as np


def normalize(X, m, s):
    """X is the numpy.ndarray of shape (d, nx) to normalize
    m is a numpy.ndarray that contains the mean of all features of X
    s is a numpy.ndarray that contains the
    standard deviation of all features of X"""
    return (X - m) / s
