# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 09:04
# @Author  : Hanchiao
# @File    : Ho-Kashyap.py

import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

class HoKashyap:
    def __init__(self):
        self.Weights = None
        self.Margins = None
        self.Samples = None

    def train(self, _samples_1, _samples_2, lr=0.3, max_iteration=100):
        """
        Ho-Kashyap training procedure
        :param _samples_1: samples coming from the class 1
        :param _samples_2: samples coming from the class 2
        :param lr: learning rate (default 0.3)
        :param max_iteration: default 100
        :return:
        """
        assert _samples_1.shape == _samples_2.shape
        # augmentation and normalization
        _homo = np.ones([_samples_1.shape[0], 1])
        _homo_s1 = np.concatenate((_samples_1, _homo), axis=1)  # augmented
        _homo = np.ones([_samples_2.shape[0], 1])
        _homo_s2 = -np.concatenate((_samples_2, _homo), axis=1)  # augmented and normalized

        self.Samples = np.concatenate((_homo_s1, _homo_s2), axis=0)  # nx3
        self.Weights = np.zeros([self.Samples.shape[1], 1])  # 3x1
        self.Margins = np.random.rand(self.Samples.shape[0], 1)  # nx1

        for _epoch in range(max_iteration):
            _error = np.matmul(self.Samples, self.Weights) - self.Margins  # nx1

            self.Margins += (lr * (_error + np.abs(_error)))
            _tmp = np.matmul(self.Samples.transpose(), self.Samples) + 1e-4 * np.eye(self.Samples.shape[1])
            _tmp = np.linalg.inv(_tmp)  # (Y'Y + e * E)^{-1}  3x3
            y_plus = np.matmul(_tmp, self.Samples.transpose())  # Y+ 3xn
            self.Weights = np.matmul(y_plus, self.Margins)
            yield _epoch, self.Margins, self.Weights.reshape(-1), np.linalg.norm(_error)**2


def hokashyap_test(visualization=True):
    # w1, w3: 0, 1, 4, 5
    # w2, w4: 2, 3, 6, 7
    _data = np.loadtxt(BASE_DIR + '/samples.csv', delimiter=',', usecols=(0, 1, 4, 5))
    data_min, data_max = _data.min(), _data.max()  # just for visualization
    omega_1 = _data[:, :2]
    omega_2 = _data[:, 2:]

    if visualization:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect(1)
        ax1.set(ylabel='$x_2$', xlabel='$x_1$', xlim=[data_min-0.5, data_max+0.5],
                ylim=[data_min-0.5, data_max+0.5])
        plt.ion()
        plt.grid(linestyle='--', linewidth=1, alpha=0.3)
        ax1.scatter(omega_1[:, 0], omega_1[:, 1], color='C0', label=r'$\omega_1$', marker='h')
        ax1.scatter(omega_2[:, 0], omega_2[:, 1], label=r"$\omega_3$", color='C1', marker='x')

    classifier = HoKashyap()
    _weights, _margins, _error = None, None, None
    _x, _y = None, None
    last_error = 0
    for _epoch, _margins, _weights, _error in classifier.train(omega_1, omega_2, lr=1):
        if _error < 1e-2:  # stop when the total error is less than 1
            break

        if np.abs(last_error - _error) < 1e-5:  # stop when converged
            print("converged!")
            break

        last_error = _error
        if visualization and _epoch % 5 == 4:
            if _weights[1] != 0:
                _x = np.linspace(data_min, data_max, 2)
                _y = -(_weights[0] * _x + _weights[2]) / _weights[1]
            decision_plane, = ax1.plot(_x, _y, color='C2', label="Decision Plane", ls='-')
            ax1.set(title='Epoch: {}'.format(_epoch+1))
            plt.legend()
            plt.savefig('../fig/Ho-Kashyap 1 iteration[%d].pdf' % (_epoch+1), bbox_inches='tight', pad_inches=0.0)
            plt.pause(0.5)
            decision_plane.remove()

    _weights /= _weights[1]
    print("Result: [{:.2f}\t{:.2f}\t{:.2f}]".format(_weights[0], _weights[1], _weights[2]))
    print("Training error: {:.2f}".format(_error))

    if _weights[1] != 0:
        _x = np.linspace(data_min, data_max, 2)
        _y = -(_weights[0] / _weights[1] * _x + _weights[2]) / _weights[1]

    ax1.plot(_x, _y, color='C2', label="Decision Plane", ls='-')
    plt.legend()
    ax1.set(title='Epoch: {}'.format(_epoch + 1))
    plt.savefig('../fig/Ho-Kashyap 1 iteration[%d].pdf' % (_epoch + 1), bbox_inches='tight', pad_inches=0.0)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    hokashyap_test(visualization=True)

