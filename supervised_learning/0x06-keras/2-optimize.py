#!/usr/bin/env python3
"""function that sets up Adam optimization for a keras model
with categorical crossentropy loss and accuracy metrics"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """network: the model to optimize
    alpha: the learning rate
    beta1: the first Adam optimization parameter
    beta2: the second Adam optimization parameter
    Returns: None"""
    model = network
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=K.optimizers.Adam(alpha, beta1, beta2))
