#!/usr/bin/env python3
"""function that calculates the determinant of a matrix"""


def determinant(matrix):
    """matrix: list of lists whose determinant should be calculated"""
    if matrix == [[]]:
        return 1
    elif not isinstance(matrix, list) or matrix == []:
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
