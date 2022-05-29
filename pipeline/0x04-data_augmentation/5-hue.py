#!/usr/bin/env python3
"""function that changes the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """image: 3D tf.Tensor containing the image to shear
    delta: amount the hue should change """
    return tf.image.adjust_hue(image, delta)
