# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:50
# @Author  : 
# @File    : DataLoader.py
import numpy as np
import pandas as pd


class DataSet:
    """ 加载数据集 """
    def __init__(self, data_root='samples.csv', split='train', eval_sample_idx=None):
        data = pd.read_csv(data_root, header=None)
        if split == 'train':
            sample_idx = [_i for _i in range(data.shape[0]) if _i not in eval_sample_idx]
        else:
            if eval_sample_idx is not None:
                sample_idx = eval_sample_idx
            else:
                sample_idx = list(range(data.shape[0]))

        labels = data.iloc[sample_idx, -1]
        features = data.iloc[sample_idx, :-1]

        labels = pd.get_dummies(labels)
        self.Labels = labels.to_numpy()
        self.Features = features.to_numpy()
        self.Num = self.Features.shape[0]

    def __len__(self):
        return self.Num

    def __getitem__(self, item):
        return self.Features[item, :], self.Labels[item, :]


class DataLoader:
    """ 数据迭代器 """
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.Dataset = dataset
        self.batch_size = batch_size
        self.Indices = np.arange(len(dataset))
        self.Num = np.floor(self.Indices.shape[0] / self.batch_size).astype(int)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.Indices)

        self.cnt = 0
        return self

    def __next__(self):
        if not self.cnt < self.Num:
            raise StopIteration
        batch_indices = self.Indices[self.cnt * self.batch_size:min((self.cnt + 1) * self.batch_size, self.Indices.shape[0])]
        self.cnt += 1
        return self.Dataset[batch_indices]

    def __len__(self):
        return self.Num


if __name__ == '__main__':
    data_root = '../samples.csv'
    training_dataset = DataSet(data_root)
    training_data_loader = DataLoader(training_dataset, 1, shuffle=True)

    from tqdm import tqdm
    from time import sleep

    for epoch in range(2):
        print('Epoch %d' % (epoch+1))
        for i, (features, target) in tqdm(enumerate(training_data_loader),
                                          total=len(training_data_loader), smoothing=0.9):
            print('features: {}\t target:{}'.format(features, target))
            sleep(0.5)