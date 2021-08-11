#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot matrix into a vector of labels"""
    try:
        inverted = np.argmax(one_hot, axis=0)
        return inverted
    except Exception:
        return None
