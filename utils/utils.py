#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import numpy as np


# short for norm
def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


# distance in poincare disk
def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)


# compute symmetric poincare distance matrix
def poincare_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = poincare_dist(embedding[i], embedding[j])
    return dist_matrix


# convert array from poincare disk to hyperboloid
def poincare_pts_to_hyperboloid(Y, eps=0, metric='minkowski'):
    X = Y.T
    d, n = np.shape(X)
    Z = np.zeros((d+1, n))
    if metric == "minkowski":
        for i in range(n):
            Z[0, i] = (1 + np.sum(X[:, i]**2)) / (1 - np.sum(X[:, i]**2) + eps)
            Z[1:, i] = 2 * X[:, i] / (1 - np.sum(X[:, i]**2) + eps)
    elif metric == "lorentz":
        for i in range(n):
            Z[d, i] = (1 + np.sum(X[:, i]**2)) / (1 - np.sum(X[:, i]**2) + eps)
            Z[0:d, i] = 2 * X[:, i] / (1 - np.sum(X[:, i]**2) + eps)
    else:
        raise NotImplementedError
    return Z.T


# define hyperboloid bilinear form
def hyperboloid_dot(u, v):
    return np.dot(u[:-1], v[:-1]) - u[-1]*v[-1]


# define alternate minkowski/hyperboloid bilinear form
def minkowski_dot(u, v):
    return u[0]*v[0] - np.dot(u[1:], v[1:]) 


# hyperboloid distance function
def hyperboloid_dist(u, v, eps=1e-6, metric='lorentz'):
    if metric == 'minkowski':
        dist = np.arccosh(-1*minkowski_dot(u, v))
    else:
        dist = np.arccosh(-1*hyperboloid_dot(u, v))
    if np.isnan(dist):
        # print('Hyperboloid dist returned nan value')
        return eps
    else:
        return dist


# compute symmetric hyperboloid distance matrix
def hyperboloid_distances(embedding):
    n = embedding.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i][j] = hyperboloid_dist(embedding[i], embedding[j])
    return dist_matrix


# project within disk
def proj(theta, eps=0.1):
    if norm(theta) >= 1:
        theta = theta/norm(theta) - eps
    return theta


# helper function to generate samples
def generate_data(n, radius=0.7, hyperboloid=False):
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, radius, n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    init_data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    if hyperboloid:
        return poincare_pts_to_hyperboloid(init_data)
    else:
        return init_data