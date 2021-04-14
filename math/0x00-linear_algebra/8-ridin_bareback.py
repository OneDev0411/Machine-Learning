#!/usr/bin/env python3
"""multiply matrices"""


def mat_mul(mat1, mat2):
    """matrix multiplication"""
    tmp = []
    mul = []
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            dot = 0
            for k in range(len(mat2)):
                dot = dot + (mat1[i][k] * mat2[k][j])
            tmp.append(dot)
        mul.append(tmp)
        tmp = []
    return mul
