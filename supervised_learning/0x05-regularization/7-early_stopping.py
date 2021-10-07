#!/usr/bin/env python3
"""that determines if you should stop gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """cost is the current validation cost
    opt_cost is the lowest recorded validation cost
    threshold is the threshold used for early stopping
    patience is the patience count used for early stopping
    count is the count of how long the threshold has not been met
    Returns: a boolean of whether the network
    should be stopped early, followed by the updated count"""
    if opt_cost <= threshold + cost:
        count += 1
    else:
        return False, 0
    if count >= patience:
        return True, count
    return False, count
