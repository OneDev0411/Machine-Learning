#!/usr/bin/env python3
"""function that performs a convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """images: numpy array with shape (m, h, w)
    containing grayscale images
    m: number of images
    h: height in pixels of the images
    w: width in pixels of the images
    kernel: numpy array with shape (kh, kw)
    containing the kernel for the convolution
    padding: tuple of (ph, pw), ‘same’, or ‘valid’
    ph: padding for the height of the image
    pw: padding for the width of the image
    stride: tuple of (sh, sw)
    sh: stride for the height of the image
    sw: stride for the width of the image
    Returns: numpy array containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        out_h = ((h - kh) // sh) + 1
        out_w = ((w - kw) // sw) + 1
        pad_h = 0
        pad_w = 0
    elif padding == 'same':
        out_h = h // sh
        out_w = w // sh
        pad_h = (((out_h - 1) * sh + kh - h) // 2) + 1
        pad_w = (((out_w - 1) * sw + kw - w) // 2) + 1
    else:
        pad_h, pad_w = padding
        out_h = (((h + (pad_h * 2)) - kh) // sh) + 1
        out_w = (((w + (pad_w * 2)) - kw) // sh) + 1
    padded = np.pad(images, ((0,), (pad_w,), (pad_h,)), 'constant')
    conv = np.zeros(shape=(m, out_h, out_w))
    for i in range(0, (out_h * out_w)):
        row = i // out_w
        col = i % out_w
        conv[:, row, col] = (
                padded[:, row * sh:kh + row*sh, col * sw:kw + col*sw]
                * kernel).sum(axis=(1, 2))
    return conv
