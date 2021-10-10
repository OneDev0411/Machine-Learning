#!/usr/bin/env python3
"""function that performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
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
    out_h = h - kh + 1
    out_w = w - kw + 1
    conv = np.zeros(shape=(m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_h):
            conv[:, i, j] = (
                    images[:, i:kh+i, j:kw+j] * kernel).sum(axis=(1, 2))
    return conv
