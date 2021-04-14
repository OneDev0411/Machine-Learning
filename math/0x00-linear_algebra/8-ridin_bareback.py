#!/usr/bin/env python3
import numpy
"""multiply matrices"""


def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    npres = numpy.matmul(mat1, mat2)
    return npres.tolist()
