#!/usr/bin/env python3
"""vanilla autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for
                        each hidden layer in the encoder, respectively
        latent_dims: integer containing the dimensions of
                    the latent space representation """
    inpt = keras.Input(shape=(input_dims,))
    for i in range(len(hidden_layers)):
        if i == 0:
            enc = (keras.layers.Dense(hidden_layers[i], activation="relu"))(inpt)
        else:
            enc = (keras.layers.Dense(hidden_layers[i], activation="relu"))(enc)
    enc = (keras.layers.Dense(latent_dims, activation="relu"))(enc)
    encoder = keras.Model(inpt, enc)

    inpt_dec = keras.Input(shape=(latent_dims,))
    dec = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(inpt_dec)
    for i in hidden_layers[-2::-1]:
        if i == len(hidden_layers) - 1:
            dec = keras.layers.Dense(
                hidden_layers[-1], activation='relu')(inpt_dec)
        else:
            dec = keras.layers.Dense(i, activation='relu')(dec)
    dec = keras.layers.Dense(input_dims, activation='sigmoid')(dec)
    decoder = keras.Model(inpt_dec, dec)

    autoen = keras.Model(inpt, decoder(encoder(inpt)))
    autoen.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, autoen
