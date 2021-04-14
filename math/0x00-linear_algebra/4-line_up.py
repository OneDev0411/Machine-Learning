#!/usr/bin/env python3
"""add arrays element wise"""


def add_arrays(arr1, arr2):
    """add arrays"""
    if len(arr1) == len(arr2):
        res = [[i + j for i, j in zip(arr1, arr2)]]
        return res
    else:
        return None
