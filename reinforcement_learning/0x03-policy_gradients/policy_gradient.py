#!/usr/bin/env python3
"""function that computes the Monte-Carlo
policy gradient based on a state and a weight matrix"""
import numpy as np


def policy(matrix, weight):
    """function that computes to policy with a weight of a matrix"""
    return np.exp(matrix.dot(weight)) / np.sum(
        np.exp(matrix.dot(weight)))


def policy_gradient(state, weight):
    """function that computes the Monte-Carlo policy
    gradient based on a state and a weight matrix"""
    policy = policy(state, weight)
    action = np.random.choice(len(policy[0]), p=policy[0])
    resh_policy = policy.reshape(-1, 1)
    softmax = (np.diagflat(resh_policy) - np.dot(
        resh_policy, resh_policy.T))[action, :]
    log = softmax / policy[0, action]
    gradient = state.T @ log[None, :]
    return action, gradient
