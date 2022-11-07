# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 10:42
# @Author  : Hanchiao
# @File    : multiclass_mse.py

import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class MulticlassMSE:
    def __init__(self, _class_num):
        self.Weights = None
        self.Labels = None
        self.ClassNum = _class_num

    def train(self, _samples, _labels):
        """
        Generalizations for MSE Procedures
        """
        _homo = np.ones([_samples.shape[0], 1])
        _samples = np.concatenate((_samples, _homo), axis=1)  # augmented nx3
        self.Labels = np.eye(self.ClassNum)[_labels]

        _tmp = np.matmul(_samples.transpose(), _samples) + 1e-3 * np.eye(_samples.shape[1])  # X'X+epsilon*E
        _tmp = np.linalg.inv(_tmp)
        x_plus = np.matmul(_tmp, _samples.transpose())
        self.Weights = np.matmul(x_plus, self.Labels)
        return self.Weights

    def predict(self, _sample):
        _sample = np.append(_sample, 1)  # augmented
        _y = np.matmul(_sample, self.Weights)
        return np.argmax(_y)

    def eval(self, _samples, _labels):
        assert _samples.shape[0] == _labels.shape[0]

        _total = _samples.shape[0]
        correct = 0
        for _idx, _s in enumerate(_samples):
            if self.predict(_s) == _labels[_idx]:
                correct += 1

        return correct / _total


def multiclass_mse_test(visualization):
    _data = np.loadtxt(BASE_DIR + '/samples.csv', delimiter=',')
    data_min, data_max = _data.min(), _data.max()
    class_num = 4
    training_data = _data[:8, :].reshape(-1, 2)
    training_gt = np.array(list(range(class_num)) * 8)

    test_data = _data[8:, :].reshape(-1, 2)
    test_gt = np.array(list(range(class_num)) * 2)

    if visualization:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect(1)
        ax1.set(ylabel='$x_2$', xlabel='$x_1$', xlim=[data_min-0.5, data_max+0.5],
                ylim=[data_min-0.5, data_max+0.5])
        plt.ion()
        plt.grid(linestyle='--', linewidth=1, alpha=0.3)
        _marker = ['h', 'v', 'o', '*']
        for _i in range(class_num):
            ax1.scatter(_data[:8, 2*_i], _data[:8, 2*_i+1], color='C%d' % _i,
                        label=r'$\omega_%d$' % (_i+1), marker=_marker[_i])

    classifier = MulticlassMSE(class_num)
    _weights = classifier.train(training_data, training_gt)
    training_accuracy = classifier.eval(training_data, training_gt)
    print("training_accuracy: {:2f}".format(training_accuracy))

    test_accuracy = classifier.eval(test_data, test_gt)
    print("test_accuracy: {:2f}".format(test_accuracy))

    if visualization:
        _weights = _weights.transpose()
        _x, _y = None, None
        for _w in _weights:
            if _w[1] != 0:
                _x = np.linspace(data_min, data_max, 3)
                _y = -(_w[0] * _x + _w[2]) / _w[1]
            decision_plane, = ax1.plot(_x, _y, color='C5', ls='-')

    plt.legend()
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    multiclass_mse_test(visualization=True)