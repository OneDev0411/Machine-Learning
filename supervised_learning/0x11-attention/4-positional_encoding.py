#!/usr/bin/env python3
""" Function that calculates the positional encoding for a transformer """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ max_seq_len: integer representing the maximum sequence length
        dm: the model depth
        Returns: numpy.ndarray of shape (max_seq_len, dm)
                containing the positional encoding vectors"""
    pos_enc = np.zeros((max_seq_len, dm))
    dmf = np.float(dm)
    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            pos_enc[i, j] = np.sin(i / np.power(10000, (2 * j // 2) / dmf))
            pos_enc[i, j + 1] = np.cos(i / np.power(10000, (2 * j // 2) / dmf))
    return pos_enc
