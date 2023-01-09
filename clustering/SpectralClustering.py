# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/9 16:43
# @Author  : hanchiao
# @File    : SpectralClustering.py
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize


def gaussian_kernel(dist, sigma=1):
    _x = dist**2/(sigma**2)/2
    return np.exp(-_x)


class SpectralClusteringNG:
    def __init__(self, n_components, affinity=None, n_neighbors=10):
        """
        :param n_components: Number of eigenvectors to use for the spectral embedding(Usually equals the number of clusters).
        :param affinity: How to construct the affinity matrix
        :param n_neighbors: Number of neighbors to use when constructing the affinity matrix using the nearest
        neighbors method.
        """
        self.Ncomp = n_components
        self.Nneigh = n_neighbors

        if affinity is not None:
            self.Affi = affinity
        else:
            self.Affi = gaussian_kernel

    def fit_tranform(self, X, kernel=gaussian_kernel):
        tree = KDTree(X, metric='euclidean')
        dist, ind = tree.query(X, k=self.Nneigh+1)  # self-included
        dist, ind = dist[:, 1:], ind[:, 1:]
        affinity = kernel(dist)
        AffiMat = self.get_AffiMat(affinity, ind)
        DegreeMat_ = np.diag(AffiMat.sum(axis=1) ** (-0.5))

        Laplacian_sym = np.eye(DegreeMat_.shape[0]) - DegreeMat_ @ AffiMat @ DegreeMat_  # L_sym = E-D^(-0.5)WD^(-0.5)

        eigenVal, eigenVec = np.linalg.eig(Laplacian_sym)  # 求取特征向量
        eigenVec = eigenVec[:, np.argsort(eigenVal)]  # sort eigenvectors(ascending)
        X_transformed = normalize(eigenVec[:, :self.Ncomp], norm='l2', axis=1)

        return X_transformed

    def get_AffiMat(self, aff, ind):
        _n = aff.shape[0]
        mat = np.zeros((_n, _n))
        for _i in range(_n):
            for _cnt, _j in enumerate(ind[_i]):
                mat[_i, _j] = aff[_i, _cnt]
        return (mat.T + mat) / 2


if __name__ == '__main__':
    import kmeans

    DATASET_DIR = 'data/spectral_clustering_samples.csv'
    DATASET = np.loadtxt(DATASET_DIR, delimiter=' ')
    X = DATASET[:, :-1]
    Y = DATASET[:, -1]
    print("Dataset loaded. {} samples in total.".format(X.shape[0]))

    # k_neighbors = [2, 4, 6, 10, 15, 20, 22, 25, 27, 30, 40, 70, 100, 190]
    # sigma = [1, 31, 63]
    # for _cnt, k in enumerate(k_neighbors):
    #     print("[Iteration {}] k={}".format(_cnt+1, k))
    #     spectral_cluster = SpectralClusteringNG(n_components=2, n_neighbors=k)
    #
    #     def kernel(dist):
    #         return gaussian_kernel(dist, sigma[2])
    #
    #     X_spectral_embedded = spectral_cluster.fit_tranform(X, kernel)
    #
    #     Kmeans = kmeans.KMeans(n_clusters=2, max_iter=1500, verbose=False)
    #     means, assign = Kmeans.fit_predict(X_spectral_embedded)
    #
    #     correct = np.sum(Y == assign)
    #     print("accuracy: {:.4f}".format(max(correct / float(Y.shape[0]), 1 - correct / float(Y.shape[0]))))

    sigma = [1, 5, 9, 13, 17, 21, 25, 29]
    k = [2, 16, 32]
    for _cnt, sig in enumerate(sigma):
        print("[Iteration {}] sigma={}".format(_cnt + 1, sig))
        spectral_cluster = SpectralClusteringNG(n_components=2, n_neighbors=k[2])

        def kernel(dist):
            return gaussian_kernel(dist, sig)

        X_spectral_embedded = spectral_cluster.fit_tranform(X, kernel)

        Kmeans = kmeans.KMeans(n_clusters=2, max_iter=1500, verbose=False)
        means, assign = Kmeans.fit_predict(X_spectral_embedded)

        correct = np.sum(Y == assign)
        print("accuracy: {:.4f}".format(max(correct / float(Y.shape[0]), 1 - correct / float(Y.shape[0]))))
