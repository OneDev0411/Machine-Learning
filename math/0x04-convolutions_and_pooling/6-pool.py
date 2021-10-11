#!/usr/bin/env python3
"""function that performs  that performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """images: numpy array with shape (m, h, w, c)
    containing grayscale images
    m: number of images
    h: height in pixels of the images
    w: width in pixels of the images
    c: the number of channels in the image
    kernel_shape: numpy array with shape (kh, kw)
    containing the shape for the pooling
    nc: the number of kernels
    stride: tuple of (sh, sw)
    sh: stride for the height of the image
    sw: stride for the width of the image
    mode indicates the type of pooling(max or avg)
    Returns: numpy array containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1
    conv = np.zeros(shape=(m, out_h, out_w, c))
    for i in range(0, (out_h * out_w)):
        row = i // out_w
        col = i % out_w
        if mode == 'max':
            conv[:, row, col, :] = (
                images[:, row * sh:kh + row * sh, col * sw:kw + col * sw, :]
            ).max(axis=(1, 2))
        elif mode == 'avg':
            conv[:, row, col, :] = np.average(
                images[:, row * sh:kh + row * sh, col * sw:kw + col * sw, :],
                axis=(1, 2))
    return conv
