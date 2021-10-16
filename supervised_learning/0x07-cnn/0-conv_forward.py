#!/usr/bin/env python3
"""function  that performs forward propagation over a
convolutional layer of a neural network"""
import numpy as np


def conv_forward(
        A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """A_prev: numpy array with shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    m: number of examples
    h_prev: height of the previous layer
    w_prev: width of the previous layer
    c_prev: the number of channels in the previous layer
    W: numpy array with shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
    kh: the filter height
    kw: the filter width
    c_prev: the number of channels in the previous layer
    c_new: the number of channels in the output
    b: numpy array with shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    activation: activation function applied to the convolution
    padding: ‘same’ or ‘valid’
    stride: tuple of (sh, sw)
    sh: stride for the height of the image
    sw: stride for the width of the image
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = (((h_prev - 1) * sh + kh - h_prev) // 2) + 1
        pad_w = (((w_prev - 1) * sw + kw - w_prev) // 2) + 1
    else:
        pad_h, pad_w = 0, 0
    out_h = ((h_prev - kh + 2 * pad_h) // sh) + 1
    out_w = ((w_prev - kw + 2 * pad_w) // sw) + 1
    padded = np.pad(A_prev, ((0,), (pad_h,), (pad_w,), (0,)), 'constant')
    conv = np.zeros(shape=(m, out_h, out_w, c_new))
    for i in range(0, (out_h * out_w)):
        row = i // out_w
        col = i % out_w
        for kernel in range(c_new):
            conv[:, row, col, kernel] = activation(((
                padded[:, row * sh:kh + row * sh, col * sw:kw + col * sw, :] *
                W[:, :, :, kernel]).sum(axis=(1, 2, 3))) + b[:, :, :, kernel])
    return conv
