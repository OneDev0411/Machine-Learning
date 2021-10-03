#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """function that updates a variable using
    the RMSProp optimization algorithm"""
    s = s * beta2 + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / (epsilon + np.sqrt(s))
    return var, s
