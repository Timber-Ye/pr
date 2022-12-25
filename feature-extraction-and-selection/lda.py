# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 19:43
# @Author  : Hanchiao
# @File    : lda.py

import numpy as np
from knn_clf import knn_clf
import scipy


class LDA:
    def __init__(self, _n_component=None):
        self.component = _n_component  # 投影后的维数
        self.Features = None
        self.Labels = None
        self.ClassNum = None  # 类别个数
        self.gEigVec = None  # 广义特征向量
        self.eigIndices = None  # 特征向量按照对应特征值由大到小排序

    def fit(self, _features, _labels):
        self.Features = _features
        mean = np.mean(self.Features, axis=0)
        self.Labels = _labels
        min_l, max_l = np.min(self.Labels), np.max(self.Labels)
        self.ClassNum = max_l - min_l + 1
        Sw = np.zeros((self.Features.shape[1], self.Features.shape[1]))  # within scatter
        St = np.cov(self.Features, rowvar=False)  # total scatter

        for _l in range(min_l, max_l+1):
            _cluster_idx = np.where(self.Labels == _l)[0]
            _cluster_n = _cluster_idx.shape[0]
            _cluster = self.Features[_cluster_idx]
            _cluster_mean = np.mean(_cluster, axis=0)

            Swi = np.cov(_cluster, rowvar=False)
            Sw += Swi

        Sb = St - Sw  # between scatter
        Sw += 1e-3 * np.eye(self.Features.shape[1])  # regularization

        eigenVal, eigenVec = scipy.linalg.eigh(Sb, Sw)
        eigenVec = eigenVec[:, np.argsort(eigenVal)[::-1]]  # sort eigenvectors
        self.gEigVec = np.array(eigenVec)

        print("Class number: {}".format(self.ClassNum))
        print("Dataset size: {:^3d}".format(self.Features.shape[0]))
        print("Dataset dim: {:^3d}".format(self.Features.shape[1]))

    def transform(self, _n_components=None):
        if _n_components is not None:
            self.component = _n_components

        assert self.component is not None
        assert self.component <= min(self.ClassNum, self.Features.shape[1])
        assert self.Features is not None

        return np.dot(self.Features, self.gEigVec[:, :self.component]), self.Labels  # 投影变换


def lda_knn_clf(_lda, _n_components=None):
    data_x_reduced, _data_y = _lda.transform(_n_components)  # LDA降维
    clf_accuracy = knn_clf(data_x_reduced, _data_y)
    return clf_accuracy


if __name__ == '__main__':
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # y = np.array([1, 1, 1, 2, 2, 2])
    # clf = LinearDiscriminantAnalysis(solver='svd', n_components=1)
    # X_new = clf.fit_transform(X, y)
    #
    # print(X_new)
    #
    # print(X)
    # print(y)
    # my_clf = LDA(1)
    # my_clf.fit(X, y)
    # my_X_new, _ = my_clf.transform()
    #
    # print(my_X_new)

    # VEHICLE_DATASET_DIR = 'data/vehicle.txt'
    # VEHICLE_DATASET = np.loadtxt(VEHICLE_DATASET_DIR, delimiter='\t', dtype=int)
    #
    # VEHICLE_DATASET_FEATURES, VEHICLE_DATASET_LABEL = VEHICLE_DATASET[:, :-1], VEHICLE_DATASET[:, -1]
    # VEHICLE_DATASET_NDIM = VEHICLE_DATASET_FEATURES.shape[1]
    #
    # my_clf = LDA(3)
    # my_clf.fit(VEHICLE_DATASET_FEATURES, VEHICLE_DATASET_LABEL)
    # my_X_new, _ = my_clf.transform()
    # #
    # clf = LinearDiscriminantAnalysis(solver='eigen', n_components=3)
    # X_new = clf.fit_transform(VEHICLE_DATASET_FEATURES, VEHICLE_DATASET_LABEL)
    # #
    # print(my_X_new)
    # print(X_new)
    # #
    # acc = knn_clf(my_X_new, VEHICLE_DATASET_LABEL)
    # print(acc)

    ORL_DATASET_DIR = 'data/ORLData_25.txt'
    ORL_DATASET = np.loadtxt(ORL_DATASET_DIR, delimiter='\t', dtype=int)
    ORL_DATASET_FEATURES, ORL_DATASET_LABEL = ORL_DATASET[:, :-1]/255, ORL_DATASET[:, -1]
    ORL_DATASET_NDIM = ORL_DATASET_FEATURES.shape[1]

    my_clf = LDA(3)
    my_clf.fit(ORL_DATASET_FEATURES, ORL_DATASET_LABEL)
    my_X_new, _ = my_clf.transform()

    # clf = LinearDiscriminantAnalysis(solver='eigen', n_components=3)
    # X_new = clf.fit_transform(ORL_DATASET_FEATURES, ORL_DATASET_LABEL)

    acc = knn_clf(my_X_new, ORL_DATASET_LABEL)
    print(acc)



