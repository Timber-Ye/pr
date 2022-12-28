# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/28 12:35
# @Author  : 
# @File    : kmeans.py
import numpy as np
import matplotlib.pyplot as plt


# def cal_mean_vector(samples):
#     """
#     Accepts a list of samples, return the mean vector of them
#     :param samples: sample points, each with the same number of dimensions.
#     :return: the vector which lies in the center of all samples.
#     """
#
#     return np.mean(samples, axis=0)


def assign_samples(samples, means):
    """
    Given a data set and a list of mean vectors,
    assign each sample to an index that corresponds to the label
    of the closest mean vector.

    :param samples: sample points, each with the same number of dimensions.
    :param means: mean vectors
    :return: A list which will mark the label of the closest mean vector for each sample.
    """
    assignments = np.zeros(samples.shape[0], dtype=int)
    for i, p in enumerate(samples):  # 遍历每一个点
        dist_min = float("inf")
        idx = 0
        for _i, v in enumerate(means):  # 计算距离最近的中心点
            dist = np.sum((v-p)**2)
            if dist < dist_min:
                dist_min = dist
                idx = _i
        assignments[i] = idx
    return assignments


def draw(_means, _X, _assign):
    plt.cla()
    n_clusters = int(np.max(_assign)) + 1
    for _i in range(n_clusters):
        cluster_idx = np.where(_assign == _i)[0]
        plt.plot(_X[cluster_idx, 0], _X[cluster_idx, 1], '.', label='cluster {}'.format(_i+1))
    plt.plot(_means[:, 0], _means[:, 1], 'r^', label='cluster centers')


def update_means(samples, assign):
    """
    Accepts a set of samples, a list of assignments, the indices
    of both lists correspond to each other.

    Compute the center for each of the assigned groups.

    :return: `k` centers where `k` is the number of clusters.
    """

    n_clusters = int(np.max(assign)) + 1
    new_means = np.zeros((n_clusters, samples.shape[1]))
    cluster_num = np.zeros(n_clusters)
    for s, l in zip(samples, assign):
        new_means[l] = new_means[l] + s
        cluster_num[l] += 1
    for i in range(n_clusters):
        new_means[i] /= cluster_num[i]

    return new_means


class KMeans:
    def __init__(self, n_clusters, max_iter=300, verbose=False):
        """
        :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
        :param max_iter: Maximum number of iterations of the k-means algorithm for a single run.
        """
        self.N = n_clusters
        self.Iter = max_iter
        self.Assignment = None
        self.Samples = None
        self.Means = None
        self.Verbose = verbose

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        :param X: sample points, each with the same number of dimensions.
        :return: A list which will mark the label of the closest mean vector for each sample.
        """
        self.Samples = X
        initial = np.random.choice(len(self.Samples), self.N, replace=False)
        self.Means = self.Samples[initial, :]  # pick N random points as cluster centers
        for i in range(self.Iter):
            self.Assignment = assign_samples(self.Samples, self.Means)
            self.Means = update_means(self.Samples, self.Assignment)

            if self.Verbose and i % 1 == 0:
                print("[Iteration {}] mean vectors:\n{}\n".format(i+1, self.Means))
                yield self.Means, self.Assignment

        print("[Output] mean vectors:\n{}\n".format(self.Means))
        return self.Means, self.Assignment

    def score(self, Y, Centers):
        """
        Compute the clustering accuracy as well as 'the mean distance error' given the ground truth,
        where 'the mean distance error' is the average distance between true centers and the cluster mean vectors.
        :param Y: ground truth of the cluster label list
        :param Centers: ground truth of the cluster centers
        :return: clustering accuracy
        """
        sorted_indices = np.argsort(self.Means[:, 0])
        dist_sum = 0
        for _i, _mean in enumerate(self.Means):
            min_dist = float('inf')
            record_idx = 0
            for _j, _c in enumerate(Centers):
                dist_sq = np.sum((_mean - _c) ** 2)
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    record_idx = _j
            sorted_indices[_i] = record_idx
            dist_sum += np.sqrt(min_dist)

        correct = np.sum(Y == sorted_indices[self.Assignment])
        return float(correct) / self.Assignment.shape[0], dist_sum / self.N

    def draw_result(self):
        draw(self.Means, self.Samples, self.Assignment)


if __name__ == '__main__':
    from scipy.stats import multivariate_normal

    Sigma = [[1, 0], [0, 1]]
    mu1 = [1, -1]
    x1 = multivariate_normal.rvs(mean=mu1, cov=Sigma, size=100)
    mu2 = [5.5, -4.5]
    x2 = multivariate_normal.rvs(mean=mu2, cov=Sigma, size=100)

    # obtain the 200 data points to be clustered
    X = np.concatenate((x1, x2))

    tmp = np.ones((100, 1))
    Y = np.concatenate((tmp, tmp*2)) - 1

    kmeans = KMeans(n_clusters=2, verbose=False, max_iter=10)
    plt.ion()
    for means, assign in kmeans.fit_predict(X):
        draw(means, X, assign)
        plt.pause(0.5)
        plt.show()

    centers = np.array([mu1, mu2])
    acc, avg_dist = kmeans.score(Y, centers)
    print("clustering accuracy: {:.2f},\t average distance: {:.2f}".format(acc, avg_dist))
    plt.ioff()
    kmeans.draw_result()
    plt.show()
