#!/usr/bin/env python3
"""function that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """nx is the number of input features
    layers is a list containing the number of nodes in each layer
    activations is a list containing the activation functions
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model"""
    model = K.Sequential()
    for i in range(len(layers)):
        if i != 0:
            model.add(K.layers.Dropout(1 - keep_prob))
        model.add(
            K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_dim=nx))
    return model
