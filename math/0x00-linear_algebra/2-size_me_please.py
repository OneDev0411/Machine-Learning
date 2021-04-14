#!/usr/bin/env python3
def matrix_shape(matrix):
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
