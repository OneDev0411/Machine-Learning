#!/usr/bin/env python3
""" function that performs K-means on a dataset """
import sklearn.cluster


def kmeans(X, k):
    """ X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
