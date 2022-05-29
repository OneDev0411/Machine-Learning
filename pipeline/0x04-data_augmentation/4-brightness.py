#!/usr/bin/env python3
"""function  that randomly changes the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """image: 3D tf.Tensor containing the image to shear
    max_delta: maximum amount the image should be brightened """
    return tf.image.adjust_brightness(image, max_delta)
