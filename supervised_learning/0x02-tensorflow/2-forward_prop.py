#!/usr/bin/env python3
"""function that creates the forward propagation graph
for the neural network:"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """x is the placeholder for the input data
    layer_sizes is a list containing the number of nodes in each layer
    activations is a list containing the activation functions for each layer
    Returns: the prediction of the network in tensor form"""
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return x
