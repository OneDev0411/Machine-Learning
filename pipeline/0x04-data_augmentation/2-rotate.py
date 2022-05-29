#!/usr/bin/env python3
"""function that rotates an image"""
import tensorflow as tf


def rotate_image(image):
    """image: 3D tf.Tensor containing the image to rotate"""
    return tf.image.rot90(image)
