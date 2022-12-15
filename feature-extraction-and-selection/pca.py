# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/13 16:51
# @Author  : 
# @File    : pca.py
import numpy as np
from knn_clf import knn_clf


class PCA:
    def __init__(self, _n_component=None):
        self.component = _n_component
        self.Features = None
        self.Center = None
        self.EigVec = None

    def fit(self, _features):
        self.Features = _features
        unbiased = self.Features - np.mean(_features, axis=0)  # 零均值化
        _, _, self.EigVec = np.linalg.svd(unbiased)
        print("Dataset size: {:^3d}".format(self.Features.shape[0]))
        print("Dataset dim: {:^3d}".format(self.Features.shape[1]))

    def transform(self, _n_components=None):
        if _n_components is not None:
            self.component = _n_components

        assert self.component is not None
        assert self.Features is not None

        return self.Features @ self.EigVec[:, :self.component]


def pca_knn_clf(_pca, _data_y, _n_components=None):
    data_x_reduced = _pca.transform(_n_components)  # PCA降维
    clf_accuracy = knn_clf(data_x_reduced, _data_y)  # KNN分类
    return clf_accuracy


if __name__ == '__main__':
    import sklearn.decomposition as skd
    VEHICLE_DATASET_DIR = 'data/vehicle.txt'
    VEHICLE_DATASET = np.loadtxt(VEHICLE_DATASET_DIR, delimiter='\t')

    VEHICLE_DATASET_FEATURES, VEHICLE_DATASET_LABEL = VEHICLE_DATASET[:, :-1], VEHICLE_DATASET[:, -1]
    VEHICLE_DATASET_NDIM = VEHICLE_DATASET_FEATURES.shape[1]

    pca = skd.PCA(18)
    X_new = pca.fit_transform(VEHICLE_DATASET_FEATURES)

    acc = knn_clf(X_new, VEHICLE_DATASET_LABEL)
    print(acc)

