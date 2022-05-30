#!/usr/bin/env python3
"""function that computes to policy with a weight of a matrix"""
import numpy as np


def policy(matrix, weight):
    """matrix: matrix
        weight: weight"""
    return np.exp(matrix.dot(weight)) / np.sum(
        np.exp(matrix.dot(weight)))
