#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Y is a numpy.ndarray containing numeric class labels
    classes is the maximum number of classes found in Y"""
    if not isinstance(classes, int):
        return None
    if not isinstance(Y, np.ndarray):
        return None
    if classes > max(Y):
        return None
    try:
        one_hot = np.zeros((Y.size, Y.max()+1))
        one_hot[np.arange(Y.size), Y] = 1
        return one_hot.T
    except Exception:
        return None
