#!/usr/bin/env python3
"""calculates the specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """confusion is a numpy.ndarray where row indices
    represent the correct labels and column indices
    represent the predicted labels"""
    tp = np.diag(confusion)
    fn = confusion.sum(axis=1) - tp
    fp = confusion.sum(axis=0) - tp
    tn = confusion.sum() - (tp + fp + fn)
    return tn / (tn + fp)
