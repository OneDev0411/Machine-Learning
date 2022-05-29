#!/usr/bin/env python3
"""function that performs a random crop of an image"""
import tensorflow as tf


def crop_image(image, size):
    """image: 3D tf.Tensor containing the image to crop"""
    return tf.image.random_crop(image, size)
