# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 15:30
# @Author  : 
# @File    : knn_clf.py

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_clf(_data_x, _data_y, _n_neighbors=1):
    # 划分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(_data_x, _data_y, test_size=0.2, random_state=26)

    clf = KNeighborsClassifier(n_neighbors=_n_neighbors)
    clf.fit(train_x, train_y)

    accuracy = clf.score(test_x, test_y)  # 测试结果
    return accuracy
