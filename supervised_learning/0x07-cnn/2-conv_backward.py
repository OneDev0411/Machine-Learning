#!/usr/bin/env python3
"""function  that performs backward propagation over a
convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """dZ: numpy array of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the
    unactivated output of the convolutional layer
        m: number of examples
        h_new: height of the output
        w_new: width of the output
        c_new: number of channels in the output
    A_prev: numpy array of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    W: numpy array of shape (kh, kw, c_prev, c_new)
    containing the kernels for the convolution
        kh is the filter height
        kw is the filter width
    b: numpy array of shape (1, 1, 1, c_new)
    containing the biases applied to the convolution
    padding: 'same' or 'valid'
    stride: tuple of (sh, sw)
        sh is the stride for the height
        sw is the stride for the width
    return: partial derivatives (dA_prev),
            the kernels (dW),
            biases (db)"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = (((h_prev - 1) * sh + kh - h_prev) // 2)
        pad_w = (((w_prev - 1) * sw + kw - w_prev) // 2)
    else:
        pad_h, pad_w = 0, 0
    A_prev_pad = np.pad(
        A_prev, ((0,), (pad_h,), (pad_w,), (0,)), 'constant')
    dW = np.zeros(shape=(kh, kw, c_prev, c_new))
    dA_prev_pad = np.zeros_like(A_prev_pad)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for im in range(m):
        for i in range(0, (h_new * w_new)):
            row = i // w_new
            col = i % w_new
            for kernel in range(c_new):
                dW[:, :, :, kernel] += A_prev_pad[
                    im, row * sh:kh + row * sh, col * sw:kw + col * sw, :
                ] * dZ[im, row, col, kernel]
                dA_prev_pad[im,
                            row * sh:kh + row * sh,
                            col * sw:kw + col * sw,
                            :] += dZ[im,
                                     row,
                                     col,
                                     kernel] * W[:,
                                                 :,
                                                 :,
                                                 kernel]
    return dA_prev_pad, dW, db
