#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """confusion is a numpy.ndarray where row indices
    represent the correct labels and column indices
    represent the predicted labels"""
    classes = np.shape(confusion)[0]
    rec = np.zeros(classes)
    for i in range(classes):
        rec[i] = np.max(confusion[i, :]) / np.sum(confusion[i, :])
    return rec
