#!/usr/bin/env python3
"""functions that tests a neural network"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """network:  the network model to test
    data: input data to test the model with
    labels: correct one-hot labels of data
    verbose: boolean that determines if output should be
    printed during the testing process
    Returns: None
    """
    network.evaluate(data, labels, verbose=verbose)
