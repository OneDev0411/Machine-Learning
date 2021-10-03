#!/usr/bin/env python3
""" momentum optimization algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the gradient descent with
    momentum optimization algorithm"""
    v = v * beta1 + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
