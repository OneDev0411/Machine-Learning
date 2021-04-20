#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

x = [i[0] for i in pca_data]
y = [i[1] for i in pca_data]
z = [i[2] for i in pca_data]


fig1 = plt.figure()
ax = Axes3D(fig1)
ax.scatter(x, y, z, c=labels, cmap='plasma')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.title("PCA of Iris Dataset")
plt.show()
