#!/usr/bin/env python3
"""Shape of matrix"""


def matrix_shape(matrix):
    if not isinstance(matrix[0], list):
        return []
    res = [len(matrix)] + matrix_shape(matrix[0])
    return res
