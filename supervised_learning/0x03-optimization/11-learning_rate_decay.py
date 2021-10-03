#!/usr/bin/env python3
"""Learning rate decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """function that updates the learning rate using inverse time decay"""
    return alpha / (1 + (decay_rate * (global_step // decay_step)))
