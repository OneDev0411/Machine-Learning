#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Y is a numpy.ndarray containing numeric class labels
    classes is the maximum number of classes found in Y"""
    try:
        one_hot = np.zeros((classes, Y.max()+1))
        one_hot[np.arange(classes), Y] = 1
        return one_hot.T
    except Exception:
        return None
