#!/usr/bin/env python3
"""functions that save and loads a model’s weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """network: the model whose weights should be saved
    filename: path of the file that the weights should be saved to
    save_format: format in which the weights should be saved
    Returns: None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """network: the model to which the weights should be loaded
    filename is the path of the file that the weights should be loaded from
    Returns: the loaded model"""
    return network.load_weights(filename)
