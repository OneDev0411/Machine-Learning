#!/usr/bin/env python3
"""function  that builds a modified
version of the LeNet-5 architecture using Keras"""
import tensorflow.keras as K


def lenet5(X):
    """X: (m, 28, 28, 1) contains input images for the network
        m: number of images
    returns: K.Model compiled to use Adam optimization"""
    kernel = K.initializers.he_normal(seed=None)
    L1_conv = K.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              kernel_initializer=kernel,
                              padding='SAME',
                              activation='relu')(X)
    L2_pool = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(L1_conv)
    L3_conv = K.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              kernel_initializer=kernel,
                              activation='relu')(L2_pool)
    L4_pool = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(L3_conv)
    L4_flat = K.layers.Flatten()(L4_pool)
    L5_fc = K.layers.Dense(units=120,
                           kernel_initializer=kernel,
                           activation='relu')(L4_flat)
    L6_fc = K.layers.Dense(units=84,
                           kernel_initializer=kernel,
                           activation='relu')(L5_fc)
    L7_fc = K.layers.Dense(units=10,
                           activation='softmax',
                           kernel_initializer=kernel)(L6_fc)
    model = K.models.Model(inputs=X, outputs=L7_fc)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
