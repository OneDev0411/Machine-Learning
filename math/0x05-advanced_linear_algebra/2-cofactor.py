#!/usr/bin/env python3
"""function that calculates the cofactor matrix of a matrix"""


def determinant(matrix):
    """matrix: list of lists whose determinant should be calculated"""
    if matrix == [[]]:
        return 1
    elif not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    elif not all(isinstance(m, list) for m in matrix):
        raise TypeError("matrix must be a list of lists")
    elif not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    width = len(matrix)
    if width == 1:
        return matrix[0][0]
    else:
        sign = -1
        det = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            sign *= -1
            det = det + sign * matrix[0][i] * determinant(m)
    return det


def minor(matrix):
    """matrix: list of lists whose minor matrix should be calculated"""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    elif not all(isinstance(m, list) for m in matrix):
        raise TypeError("matrix must be a list of lists")
    elif not all(len(row) == len(matrix) for row in matrix
                 ) or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    width = len(matrix)
    if width == 1:
        return [[1]]
    minr = []
    for i in range(width):
        buff = []
        for j in range(width):
            sliced = [row[:] for row in matrix]
            del sliced[i]
            for col in sliced:
                del col[j]
            buff.append(determinant(sliced))
        minr.append(buff)
    return minr


def cofactor(matrix):
    minor_matrix = minor(matrix)
    width = len(matrix)
    if width == 1:
        return [[1]]
    for i in range(width * width):
        row = i // width
        col = i % width
        sign = 1
        if (row + col) % 2 != 0:
            sign *= -1
        minor_matrix[row][col] *= sign
    return minor_matrix
