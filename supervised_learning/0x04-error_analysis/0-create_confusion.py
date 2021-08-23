#!/usr/bin/env python3
"""creates a confusion matrix"""
import numpy
import numpy as np


def create_confusion_matrix(labels, logits):
    """labels contains the correct labels for each data point
    logits contains the predicted labels"""
    m, classes = np.shape(labels)
    cm = np.zeros((classes, classes))
    for i in range(m):
        cm[np.where(labels[i, :] == 1), np.where(logits[i, :] == 1)] += 1
    return cm
