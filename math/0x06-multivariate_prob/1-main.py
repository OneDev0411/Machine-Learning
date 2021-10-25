#!/usr/bin/env python3

import numpy as np#!/usr/bin/env python3

import numpy as np
correlation = __import__('1-correlation').correlation

try:
    correlation(np.array([1, 2, 3, 4]))
except ValueError as e:
    print(str(e))
try:
    correlation(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
except ValueError as e:
    print(str(e))