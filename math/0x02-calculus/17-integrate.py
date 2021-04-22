#!/usr/bin/env python3
"""Integrate"""
def poly_integral(poly, C=0):
    """ Calculate the integral of a polynomial """
    if not isinstance(C, (float,int)):
        return None
    if not isinstance(poly, list) or poly == []:
        return None
    if poly[0] == 0 and len(poly) == 1:
        return [C]
    integral_poly = [C]
    for i in range(len(poly)):
        if not isinstance(poly[i],(float,int)):
            return None
        a = poly[i]/(i+1)
        if a % 1 == 0:
            integral_poly.append(int(a))
        else:
            integral_poly.append(a)
    return integral_poly
