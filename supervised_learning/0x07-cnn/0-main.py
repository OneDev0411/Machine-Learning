#!/usr/bin/env python3

import numpy as np
conv_forward = __import__('0-conv_forward').conv_forward

np.random.seed(1)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
cin = np.random.randint(2, 5)
cout = np.random.randint(5, 10)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

X = np.random.uniform(size=(m, h, w, cin))
W = np.random.uniform(size=(fh, fw, cin, cout))
b = np.random.uniform(size=(1, 1, 1, cout))
activation = lambda x: np.maximum(x, 0)
Y = conv_forward(X, W, b, activation, padding="valid", stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
print(Y[50:55])
print(Y.shape)
