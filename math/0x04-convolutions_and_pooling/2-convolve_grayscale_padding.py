#!/usr/bin/env python3
"""function that performs a convolution on
grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """images: numpy array with shape (m, h, w)
    containing grayscale images
    m: number of images
    h: height in pixels of the images
    w: width in pixels of the images
    kernel: numpy array with shape (kh, kw)
    containing the kernel for the convolution
    padding: tuple of (ph, pw)
    ph: padding for the height of the image
    pw: padding for the width of the image
    Returns: numpy array containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    out_h = (h + (pad_h * 2)) - kh + 1
    out_w = (w + (pad_w * 2)) - kw + 1
    conv = np.zeros(shape=(m, out_h, out_w))
    padded = np.zeros(shape=(m, h + (pad_h * 2), w + (pad_w * 2)))
    padded[:, pad_h:-pad_h, pad_w:-pad_w] = images
    for i in range(0, (h * w)):
        row = i // w
        col = i % w
        conv[:, row, col] = (
            padded[:, row:kh + row, col:kw + col] * kernel).sum(axis=(1, 2))
    return conv
