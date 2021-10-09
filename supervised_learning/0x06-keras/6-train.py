#!/usr/bin/env python3
"""function that hat trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        verbose=True,
        shuffle=False):
    """network is the model to train
    data: array containing the input data
    labels: is a one-hot containing the labels of data
    batch_size: size of the batch used for mini-batch gradient descent
    epochs: the number of passes through data
    validation_data: data to validate the model with, if not None
    early_stopping: boolean indicates whether early stopping
    should be used
    patience is the patience used for early stopping
    verbose: boolean that determines if output should be
    printed during training
    shuffle: boolean that determines whether to shuffle the
    batches every epoch.
    Returns: History object generated after training the model"""
    model = network
    if validation_data and early_stopping is not True:
        callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
    else:
        callback = None
    hist_obj = model.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[callback],
        validation_data=validation_data,
        shuffle=shuffle,
    )
    return hist_obj
