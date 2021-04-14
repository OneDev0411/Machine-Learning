#!/usr/bin/env python3
"""add matrices element wise"""


def add_matrices2D(mat1, mat2):
    if matrix_shape(mat1) == matrix_shape(mat2):
        res = [[mat1[i][j]+mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
        return res
    else:
        return None


def matrix_shape(matrix):
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
