#!/usr/bin/env python3
"""functions that makes a prediction using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """network:  the network model to make the prediction with
    data: input data to make the prediction with
    verbose: boolean that determines if output should be
    printed during the prediction process
    Returns: prediction for the data
    """
    return network.predict(data, verbose=verbose)
