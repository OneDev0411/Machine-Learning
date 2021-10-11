#!/usr/bin/env python3

import numpy as np
convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding

np.random.seed(2)
m = np.random.randint(1000, 2000)
h, w = np.random.randint(100, 200, 2).tolist()
fh, fw = (np.random.randint(3, 10, 2)).tolist()

images = np.random.randint(0, 256, (m, h, w))
kernel = np.random.randint(0, 10, (fh, fw))
conv_ims = convolve_grayscale_padding(images, kernel, (0, 0))
print(conv_ims)
print(conv_ims.shape)
