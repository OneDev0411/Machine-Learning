#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """data and lambtha"""
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        e = 2.7182818285
        factorial_k = 1
        if k >= 1:
            for i in range(1, k + 1):
                factorial_k = factorial_k * i
        return (e ** (- self.lambtha)) * (self.lambtha ** k) / factorial_k

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k+1):
            cdf += self.pmf(i)
        return cdf
