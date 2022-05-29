#!/usr/bin/env python3
"""function that randomly shears an image"""
import tensorflow as tf


def shear_image(image, intensity):
    """image: 3D tf.Tensor containing the image to shear
    intensity: intensity with which the image should be sheared"""
    return tf.keras.preprocessing.image.random_shear(
        image.numpy(),
        intensity,
        channel_axis=2)
