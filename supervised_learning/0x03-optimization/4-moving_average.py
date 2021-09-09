#!/usr/bin/env python3
"""function that that calculates
the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """data is the list of data to calculate the moving average of
    beta is the weight used for the moving average"""
    weighted = []
    weight = 0
    for i in range(len(data)):
        weight = ((1 - beta) * data[i]) + (beta * weight)
        weighted.append(weight / (1 - beta**(i+1)))
    return weighted
