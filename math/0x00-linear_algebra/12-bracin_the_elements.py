#!/usr/bin/env python3
"""add, sub, mul and division element wise of np-arr"""


def np_elementwise(mat1, mat2):
    """add, sub, mul and division element wise of np-arr"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
