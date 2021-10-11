#!/usr/bin/env python3
"""function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """images: numpy array with shape (m, h, w)
    containing grayscale images
    m: number of images
    h: height in pixels of the images
    w: width in pixels of the images
    kernel: numpy array with shape (kh, kw)
    containing the kernel for the convolution
    Returns: numpy array containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv = np.zeros(shape=(m, h, w))
    padded = np.zeros(shape=(m, h+2, w+2))
    padded[:, 1:-1, 1:-1] = images
    for i in range(0, (h * w)):
        row = i // h
        col = i % w
        conv[:, row, col] = (
            padded[:, row:kh + row, col:kw + col] * kernel).sum(axis=(1, 2))
    return conv
