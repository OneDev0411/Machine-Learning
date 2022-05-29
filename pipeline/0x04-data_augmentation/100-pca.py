#!/usr/bin/env python3
"""function  that performs PCA color augmentation
as described in the AlexNet paper"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """image: 3D tf.Tensor containing the image to change
        alphas: tuple of length 3 """
    img = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    img = img.astype('float32')
    img -= np.mean(img)
    img /= np.std(img)
    lambdas, a = np.linalg.eig(np.cov(img, rowvar=False))
    delta = np.dot(a, alphas * lambdas)
    pca_aug = img + delta
    pca = pca_aug * std + mean
    pca = pca.reshape(image.shape[0], image.shape[1], 3)
    pca = np.maximum(np.minimum(pca, 255), 0).astype('uint8')
    return pca
