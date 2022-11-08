# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:50
# @Author  : 
# @File    : DataLoader.py
import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, data_root='samples.csv', sample_idx=None):
        data = pd.read_csv(data_root, header=None)
        if sample_idx is not None:
            labels = data.iloc[sample_idx, -1]
            features = data.iloc[sample_idx, :-1]

        else:
            labels = data.iloc[:, -1]
            features = data.iloc[:, :-1]

        labels = pd.get_dummies(labels)
        self.Labels = labels.to_numpy()
        self.Features = features.to_numpy()
        self.Num = self.Features.shape[0]

    def __len__(self):
        return self.Num

    def __getitem__(self, item):
        return self.Features[item, :], self.Labels[item, :]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.Dataset = dataset
        self.batch_size = batch_size
        self.Indices = np.arange(len(dataset))
        self.Num = np.floor(self.Indices.shape[0] / self.batch_size).astype(int)
        if shuffle:
            np.random.shuffle(self.Indices)

    def __getitem__(self, item):
        batch_indices = self.Indices[item * self.batch_size:min((item + 1) * self.batch_size, self.Indices.shape[0])]
        return self.Dataset[batch_indices]

    def __len__(self):
        return self.Num
