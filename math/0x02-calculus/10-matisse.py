#!/usr/bin/env python3
"""Calculates the derivate of a polynomial"""
def poly_derivative(poly):
    """Calculates the derivate of a poly"""
    if not isinstance(poly, list) or poly == []:
        return None
    deriv = [poly[i] * i for i in range(1, len(poly))]
    if len(poly) == 1:
        deriv = [0]
    return deriv
