#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """confusion is a numpy.ndarray where row indices
    represent the correct labels and column indices
    represent the predicted labels"""
    rec = np.diag(confusion) / confusion.sum(axis=1)
    return rec
