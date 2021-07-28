#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """n number of trials, p probability of a success"""
        self.data = data
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
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
            ns = 0
            for i in data:
                ns = ns + ((i - mean) ** 2)
            self.n = round(mean ** 2 / (mean - (ns / len(data))))
            self.p = float(mean / self.n)

    @staticmethod
    def fac(k):
        """ factorial """

        fac = 1
        for i in range(2, k + 1):
            fac = fac * i
        return fac

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        return (self.fac(self.n) / (self.fac(k) * self.fac(self.n - k))
                ) * (self.p ** k) * ((1 - self.p) ** (self.n - k))
