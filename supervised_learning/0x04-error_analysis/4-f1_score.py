#!/usr/bin/env python3
"""calculates the F1 score for each class in a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """confusion is a numpy.ndarray where row indices
    represent the correct labels and column indices
    represent the predicted labels"""
    rec = sensitivity(confusion)
    prec = precision(confusion)
    return (2 * rec * prec) / (rec + prec)
