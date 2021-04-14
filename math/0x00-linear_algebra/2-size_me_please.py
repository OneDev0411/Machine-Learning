#!/usr/bin/env python3
"""shape of matrix"""


def matrix_shape(matrix):
    if not isinstance(matrix, list):
        return []
    res = [len(matrix)] + matrix_shape(matrix[0])
    return res
