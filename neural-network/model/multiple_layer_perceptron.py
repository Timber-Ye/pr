# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:05
# @Author  : Hanchiao
# @File    : multiple_layer_perceptron.py

from . import nnlayers


class get_model:
    def __init__(self, _input_dim, _output_dim, _num_hidden):
        self.linear1 = nnlayers.Linear(_input_dim, _num_hidden)
        self.linear2 = nnlayers.Linear(_num_hidden, _output_dim)
        self.tanh = nnlayers.Tanh()
        self.sigmoid = nnlayers.Sigmoid()

        self.z_2 = None
        self.net_2 = None
        self.z_1 = None
        self.net_1 = None
        self.x = None

    def forward(self, _x):
        self.x = _x
        self.net_1 = self.linear1(self.x)
        self.z_1 = self.tanh(self.net_1)
        self.net_2 = self.linear2(self.z_1)
        self.z_2 = self.sigmoid(self.net_2)

        y = self.z_2.reshape(self.z_2.shape[0])

        return y

    def backward(self, _error):
        delta_1 = self.sigmoid.backward(self.net_2, _error)
        delta_2 = self.linear2.backward(self.z_1, delta_1)
        delta_3 = self.tanh.backward(self.net_1, delta_2)
        delta_4 = self.net_1(self.x, delta_3)

    def update(self, _lr):
        self.linear1.update(_lr)
        self.linear2.update(_lr)

    def __str__(self):
        return "[Input Layer to Hidden Layer]\n {}\n\n[Hidden Layer to Output Layer]".format(self.linear1, self.linear2)

    def state_dict(self):
        state = {
            'Linear_1': self.linear1.state_dict(),
            'Linear_2': self.linear2.state_dict(),
        }
        return state
