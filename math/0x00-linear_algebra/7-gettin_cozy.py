#!/usr/bin/env python3
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    if len(mat1[0]) != len(mat2):
        return None
    npmat1 = np.array(mat1)
    npmat2 = np.array(mat2)
    npres = np.concatenate((npmat1, npmat2), axis=axis)
    res = npres.tolist()
    return res
