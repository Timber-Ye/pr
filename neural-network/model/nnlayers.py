# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 11:33
# @Author  : Hanchiao
# @File    : nnlayers.py
import numpy as np


class Linear:
    def __init__(self, _in_dim, _out_dim, _init_w=None, _init_b=None):
        if _init_w is not None:
            self.Weights = _init_w
        else:
            self.Weights = np.random.normal(0, 0.1, (_in_dim, _out_dim))

        if _init_b is not None:
            self.Bias = _init_b
        else:
            self.Bias = np.random.normal(0, 0.1, (1, _out_dim))

        self._grad_weights = np.zeros((_in_dim, _out_dim))
        self._grad_bias = np.zeros((1, _out_dim))

    def forward(self, _x):
        return _x@self.Weights + self.Bias  # operands broadcast

    def backward(self, _x, _error):
        self._grad_weights = _x.T @ _error
        self._grad_bias = _error.T @ np.ones(_x.shape[0])
        return _error @ self.Weights.T

    def update(self, _lr):
        self.Weights += self._grad_weights * _lr
        self.Bias += self._grad_bias * _lr

    def __call__(self, _net):
        return self.forward(_net)

    def __str__(self):
        return "Weights: {}\n Bias: {}\n".format(self.Weights, self.Bias)

    def state_dict(self):
        state = {
            'Weights': self.Weights,
            'Bias': self.Bias,
            'Grad_Weights': self._grad_weights,
            'Grad_Bias': self._grad_bias,
        }
        return state


class Relu:
    def __init__(self):
        pass

    def forward(self, _net):
        return np.where(_net < 0, 0, _net)

    def backward(self, _net, _error):
        return np.where(_net < 0, 0, 1) * _error

    def __call__(self, _net):
        return self.forward(_net)


class Tanh:
    def __init__(self):
        self.f = None

    def forward(self, _net):
        self.f = np.tanh(_net)
        return self.f

    def backward(self, _net, _error):
        return (1 - self.f ** 2) * _error

    def __call__(self, _net):
        return self.forward(_net)


class Sigmoid:
    def __init__(self):
        self.f = None

    def forward(self, _net):
        self.f = 1 / (1 + np.exp(-_net))
        return self.f

    def backward(self, _net, _error):
        return self.f * (1 - self.f) * _error

    def __call__(self, _net):
        return self.forward(_net)