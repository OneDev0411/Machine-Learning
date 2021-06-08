#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """data, mean and stddev"""
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            squared_diff = 0
            for i in range(len(data)):
                squared_diff += (data[i] - self.mean) ** 2
            self.stddev = (squared_diff / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean
