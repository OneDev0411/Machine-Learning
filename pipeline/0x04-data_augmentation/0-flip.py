#!/usr/bin/env python3
"""function  that flips an image horizontally"""
import tensorflow as tf


def flip_image(image):
    """image: 3D tf.Tensor containing the image to flip"""
    return tf.image.flip_left_right(image)
