#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """n is the number of Bernoulli trial, p is the probability of a success"""
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not(p < 1 or p > 0):
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            self.n = round(len(data) / 2)
            self.p = float(mean / self.n)
