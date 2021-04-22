#!/usr/bin/env python3
"""sum of i squared(n)"""


def summation_i_squared(n):
    """sum of i squared of n"""
    if isinstance(n, int) and n >= 1:
        return int((n * (n + 1) * (2 * n + 1)) / 6)
    else:
        return None
