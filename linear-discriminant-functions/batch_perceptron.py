# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 09:56
# @Author  : Hanchiao
# @File    : batch_perceptron.py

import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


class BatchPerceptron:
    def __init__(self):
        self.Weights = None  # the separating vector
        self.Samples = None

    def train(self, _samples_1, _samples_2, lr=1.0, max_iteration=100):
        """
        Batch Perceptron training procedure
        :param _samples_1: samples coming from the class 1
        :param _samples_2: samples coming from the class 2
        :param lr: learning rate (default 1)
        :param max_iteration: (default 100)
        :return:
        """
        assert _samples_1.shape == _samples_2.shape

        # augmentation and normalization
        _homo = np.ones([_samples_1.shape[0], 1])
        _homo_s1 = np.concatenate((_samples_1, _homo), axis=1)  # augmented
        _homo = np.ones([_samples_2.shape[0], 1])
        _homo_s2 = -np.concatenate((_samples_2, _homo), axis=1)  # augmented and normalized

        self.Samples = np.concatenate((_homo_s1, _homo_s2), axis=0)
        self.Weights = np.zeros(_samples_1.shape[1] + 1)

        for _epoch in range(max_iteration):
            incorrect = [_i for _i in self.Samples if np.dot(_i, self.Weights) <= 0]
            incorrect = np.array(incorrect)
            yield _epoch, self.Weights, incorrect.shape[0]
            step = lr * np.sum(incorrect, axis=0)
            self.Weights += step


def batch_perceptron_test(visualization=True):
    # w1, w2: 0, 1, 2, 3
    # w3, w2: 4, 5, 2, 3
    _data = np.loadtxt(BASE_DIR+'/samples.csv', delimiter=',', usecols=(4, 5, 2, 3))
    data_min, data_max = _data.min(), _data.max()  # just for visualization
    omega_1 = _data[:, :2]  # nx2
    omega_2 = _data[:, 2:]

    if visualization:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect(1)
        ax1.set(ylabel='$x_2$', xlabel='$x_1$', xlim=[data_min-0.5, data_max+0.5],
                ylim=[data_min-0.5, data_max+0.5])
        plt.ion()
        plt.tight_layout()
        plt.grid(linestyle='--', linewidth=1, alpha=0.3)
        ax1.scatter(omega_1[:, 0], omega_1[:, 1], color='C0', label=r'$\omega_3$', marker='h')
        ax1.scatter(omega_2[:, 0], omega_2[:, 1], label=r"$\omega_2$", color='C1', marker='x')

    classifier = BatchPerceptron()  # initialize
    _weights, epoch = None, None
    _x, _y = None, None
    for epoch, _weights, incorrect_num in classifier.train(omega_1, omega_2, max_iteration=100, lr=0.1):
        if incorrect_num < 1:  # stop when all have been correctly classified.
            break

        if visualization and epoch % 5 == 4:
            if _weights[1] != 0:
                _x = np.linspace(data_min, data_max, 2)
                _y = -(_weights[0] * _x + _weights[2]) / _weights[1]
                decision_plane, = ax1.plot(_x, _y, color='C2', label="Decision Plane", ls='-')
                ax1.set(title='Epoch: {}'.format(epoch))
                plt.legend()
                plt.pause(0.5)
                plt.savefig('../fig/Batch Perceptron 2 iteration[%d].pdf' % epoch, bbox_inches='tight', pad_inches=0.0)
                decision_plane.remove()

    _weights /= _weights[1]
    print("Result: [{:.2f}\t{:.2f}\t{:.2f}]".format(_weights[0], _weights[1], _weights[2]))
    print("numer of iterations: {}".format(epoch))

    if _weights[1] != 0:
        _x = np.linspace(data_min, data_max, 2)
        _y = -(_weights[0] / _weights[1] * _x + _weights[2]) / _weights[1]

    ax1.plot(_x, _y, color='C2', label="Decision Plane", ls='-')
    plt.legend()
    ax1.set(title='Epoch: {}'.format(epoch))
    plt.savefig('../fig/Batch Perceptron 2 iteration[%d].pdf' % epoch, bbox_inches='tight', pad_inches=0.0)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    batch_perceptron_test(visualization=True)


