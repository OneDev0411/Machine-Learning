#!/usr/bin/env python3
"""function that hat trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """network is the model to train
    data: array containing the input data
    labels: is a one-hot containing the labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: the number of passes through data
    verbose: boolean that determines if output should be printed during training
    shuffle: boolean that determines whether to shuffle the batches every epoch.
    Returns: History object generated after training the model"""
    model = network
    hist_obj = model.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
    )
    return hist_obj
