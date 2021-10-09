#!/usr/bin/env python3
"""functions that save and loads a model’s configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """network: the model whose configuration should be saved
    filename: path of the file that the configuration should be saved to
    Returns: None
    """
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """network: the model to which the weights should be loaded
    filename is the path of the file that the configuration should be loaded from
    Returns: the loaded model"""
    with open(filename, "r") as f:
        conf = f.read()
    return K.models.model_from_json(conf)
