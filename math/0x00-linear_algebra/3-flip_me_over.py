#!/usr/bin/env python3
"""transpose of matrix"""
def matrix_transpose(matrix):
    trans = [[0 for i in range(len(matrix))] for j in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            trans[j][i] = matrix[i][j]
    return trans
