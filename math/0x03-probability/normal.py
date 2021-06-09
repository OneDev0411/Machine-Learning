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
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            squared_diff = 0
            for i in range(len(data)):
                squared_diff += (data[i] - self.mean) ** 2
            self.stddev = (squared_diff / len(data)) ** 0.5

    @staticmethod
    def erf(x):
        """Gauss error function"""
        pi = 3.1415926536
        return (2 / (pi ** 0.5)) * (x - x ** 3 / 3 +
                                    x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the PDF of a given x-value"""
        pi = 3.1415926536
        e = 2.7182818285
        return (1 / (self.stddev * ((2 * pi) ** 0.5))) * \
            (e ** (-0.5 * (((x - self.mean) / self.stddev) ** 2)))

    def cdf(self, x):
        """Calculates the CDF of a given x-value"""
        return 0.5 * (1 + self.erf((x - self.mean) /
                      (self.stddev * (2 ** 0.5))))
