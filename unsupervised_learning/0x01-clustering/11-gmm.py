#!/usr/bin/env python3
""" function that performs K-means on a dataset """
import sklearn.mixture


def gmm(X, k):
    """ X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters """
    model = sklearn.mixture.GaussianMixture(k).fit(X)
    clss = model.predict(X)
    return model.weights_, model.means_, model.covariances_, clss, model.bic(X)
