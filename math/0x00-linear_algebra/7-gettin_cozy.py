#!/usr/bin/env python3
"""concat 2 matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concat 2 matrices along a specific axis"""
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat = zip(mat1, mat2)
        return [i + j for i, j in cat]
    else:
        if len(mat1[0]) != len(mat2[0]):
            return None
        cat = [res.copy() for res in mat1] + [res.copy() for res in mat2]
        return cat
