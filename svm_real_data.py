#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.
# Copyright 2021 Chao Pan.

import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from numpy.linalg import norm
from sklearn.svm import LinearSVC
from GrahamScan import GrahamScan
import matplotlib.pyplot as plt
from platt import *
import time
from hsvm import *
import argparse
from algos import ConvexHull,minDpair,Weightedmidpt
plt.style.use('seaborn')

"""
This is the code for our SVM experiments on real data. Data is assumed to be in folder 'embedding'. You can change it to other directories if needed.
"""

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
          'tab:olive']


def Mobius_add(x, y, c=1.):
    DeNom = 1 + 2*c*np.dot(x, y) + (c**2)*(norm(x)**2)*(norm(y)**2)
    Nom = (1 + 2*c*np.dot(x, y) + c*(norm(y)**2))*x + (1 - c*(norm(x)**2))*y
    return Nom/DeNom


def Exp_map(v, p, c=1.):
    lbda = 2/(1-c*np.dot(p,p))
    temp = np.tanh(np.sqrt(c)*lbda*np.sqrt(np.dot(v, v))/2)*v/np.sqrt(c)/np.sqrt(np.dot(v, v))
    return Mobius_add(p, temp,c)


def Log_map(x, p, c=1.):
    lbda = 2/(1-c*np.dot(p,p))
    temp = Mobius_add(-p, x, c)
    return 2/np.sqrt(c)/lbda * np.arctanh(np.sqrt(c)*norm(temp)) * temp / norm(temp)


def poincare_dist(x, y):
    return np.arccosh(1 + 2*(norm(x-y)**2)/(1-norm(x)**2)/(1-norm(y)**2))


def point_on_geodesic(x, y, t):
    return Exp_map(t*Log_map(y, x), x)


def zero_based_labels(y):
    labels = list(np.unique(y))
    new_y = [labels.index(y_val) for y_val in y]
    return np.array(new_y)


def plot_geodesic_old(p, v):
    # p is the reference point and v is the speed vector perpendicular to w
    max_R = 0.999
    t = np.linspace(0, 3, 500)
    geo_line = np.zeros((500, 2))
    count = 0
    for i in range(500):
        if t[i] == 0:
            tmp = p
        else:
            tmp = Exp_map(v*t[i], p)
        if norm(tmp) > max_R:
            break
        geo_line[count, :] = tmp
        count += 1
    return geo_line[0: count, :]


def plot_geodesic_new(p0, v, ax, c):
    R = 0.999
    t = np.linspace(0, 1, 100)
    # pos
    Line = np.zeros((2, 100))
    for n in range(1, 100):
        Line[:, n] = Exp_map(v * t[n], p0)
    Line[:, 0] = p0
    AdLine = np.zeros((2, 100))
    count = 1.0
    while np.linalg.norm(Line[:, -1]) < R:
        for n in range(100):
            AdLine[:, n] = Exp_map(v * (t[n] + count), p0)
        Line = np.append(Line, AdLine, axis=1)
        count += 1.
    ax.plot(Line[0, :], Line[1, :], c=c)
    # neg
    Line = np.zeros((2, 100))
    for n in range(1, 100):
        Line[:, n] = Exp_map(-v * t[n], p0)
    Line[:, 0] = p0
    count = 1.0
    while np.linalg.norm(Line[:, -1]) < R:
        for n in range(100):
            AdLine[:, n] = Exp_map(-v * (t[n] + count), p0)
        Line = np.append(Line, AdLine, axis=1)
        count += 1.
    ax.plot(Line[0, :], Line[1, :], c=c)


def Eval(X, y, y_pred, p=None, w1=None, xi=None, C=None):
    if p is not None:
        d = np.shape(X)[0]
        N = np.shape(X)[1]
        lmda_p = 2 / (1 - np.linalg.norm(p) ** 2)
        Z = np.zeros((d, N))
        I = np.identity(d)
        for n in range(N):
            vn = Log_map(X[:, n], p)
            etan = 2 * np.tanh(lmda_p * np.linalg.norm(vn) / 2) / np.linalg.norm(vn) / (
                        1 - np.tanh(lmda_p * np.linalg.norm(vn) / 2) ** 2)
            Z[:, n] = etan * vn

        y_hat1 = np.zeros(N)
        y_hat2 = np.zeros(N)
        decision_val = np.zeros(N)
        for n in range(N):
            if (w1 is not None):
                y_hat1[n] = np.sign(np.dot(w1, Log_map(X[:, n], p)))
                if y_hat1[n] != y[n]:
                    print(n, y_hat1[n], y[n])
                decision_val[n] = np.dot(w1, Log_map(X[:, n], p))

        if (w1 is not None):
            return np.sum(y * y_hat1 > 0) / N * 100, decision_val


def tangent_hsvm(X_train, train_labels, X_test, test_labels, C, p=None, p_arr=None, multiclass=False, curvature_const=1.0):
    # the labels need to be 0-based indexed
    start = time.time()
    n_classes = train_labels.max() + 1
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    if multiclass:
        # there is more than 2 classes, using ovr strategy
        # find optimal p for each ovr classifier
        test_probability = np.zeros((n_test_samples, n_classes), dtype=float)
        for class_label in range(n_classes):
            # print('Processing class:', class_label)
            pos_coords = []
            neg_coords = []
            binarized_labels = []
            for i in range(n_train_samples):
                if train_labels[i] == class_label:
                    pos_coords.append((X_train[i][0], X_train[i][1]))
                    binarized_labels.append(1)
                else:
                    neg_coords.append((X_train[i][0], X_train[i][1]))
                    binarized_labels.append(-1)
            if len(pos_coords) <= 1:
                # skip very small classes
                continue
            binarized_labels = np.array(binarized_labels)
            # # convex hull of positive cluster
            # pos_hull = GrahamScan(pos_coords)
            # neg_hull = GrahamScan(neg_coords)
            # print('Convex hull generated!')
            # # get the reference point p by finding the min dis pair
            # p = np.zeros(2)
            # min_dis = float('inf')
            # break_flag = False
            # for i in range(pos_hull.shape[0]):
            #     for j in range(neg_hull.shape[0]):
            #         if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
            #             min_dis = poincare_dist(pos_hull[i], neg_hull[j])
            #             p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
            #             if min_dis < 1e-2:
            #                 break_flag = True
            #                 break
            #         print('\r{}/{}, {}/{}, {}'.format(i, pos_hull.shape[0], j, neg_hull.shape[0], min_dis), end='')
            #     if break_flag:
            #         break
            # print('\nreference point p found!')
            p = p_arr[class_label]
            # map training data using log map
            X_train_log_map = np.zeros_like(X_train, dtype=float)
            for i in range(n_train_samples):
                X_train_log_map[i] = Log_map(X_train[i], p, curvature_const)
            # print('log transformation done!')
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=False)
            linear_svm.fit(X_train_log_map, binarized_labels)
            # print('binary SVM done!')
            w = linear_svm.coef_[0]
            decision_vals = np.array([np.dot(w, x) for x in X_train_log_map])
            ab = SigmoidTrain(deci=decision_vals, label=binarized_labels, prior1=None, prior0=None)
            # print('Platt probability computed!')
            # map testing data using log map
            for i in range(n_test_samples):
                x_test_log_map = Log_map(X_test[i], p, curvature_const)
                test_decision_val = np.dot(w, x_test_log_map)
                test_probability[i, class_label] = SigmoidPredict(deci=test_decision_val, AB=ab)
            # print('Probability for each test samples computed!')
        y_pred = np.argmax(test_probability, axis=1)
    else:
        # if there if only two classes, no need for Platt probability
        # if p is given, use the given p, else first estimate p
        if p is None:
            pos_coords = []
            neg_coords = []
            for i in range(n_train_samples):
                if train_labels[i] == 1:
                    pos_coords.append((X_train[i][0], X_train[i][1]))
                else:
                    neg_coords.append((X_train[i][0], X_train[i][1]))
            # convex hull of positive cluster
            pos_hull = GrahamScan(pos_coords)
            neg_hull = GrahamScan(neg_coords)
            # get the reference point p by finding the min dis pair
            p = np.zeros(2)
            min_dis = float('inf')
            for i in range(pos_hull.shape[0]):
                for j in range(neg_hull.shape[0]):
                    if poincare_dist(pos_hull[i], neg_hull[j]) < min_dis:
                        min_dis = poincare_dist(pos_hull[i], neg_hull[j])
                        p = point_on_geodesic(pos_hull[i], neg_hull[j], 0.5)
        # we have p now
        X_train_log_map = np.zeros_like(X_train, dtype=float)
        for i in range(n_train_samples):
            X_train_log_map[i] = Log_map(X_train[i], p, curvature_const)
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=False)
        linear_svm.fit(X_train_log_map, train_labels)
        X_test_log_map = np.zeros_like(X_test, dtype=float)
        for i in range(n_test_samples):
            X_test_log_map[i] = Log_map(X_test[i], p, curvature_const)
        y_pred = linear_svm.predict(X_test_log_map)
    return accuracy_score(test_labels, y_pred)*100, time.time() - start


def cho_hsvm(X_train, train_labels, X_test, test_labels, C, multiclass=False, max_epoches=30):
    # fit multiclass hsvm and get prediction accuracy
    start = time.time()
    n_train_samples = X_train.shape[0]
    hsvm_clf = LinearHSVM(early_stopping=2, C=C, num_epochs=max_epoches, lr=0.001, verbose=True,
                          multiclass=multiclass, batch_size=int(n_train_samples/50))
    hsvm_clf.fit(poincare_pts_to_hyperboloid(X_train, eps=1e-6, metric='minkowski'), train_labels)
    y_pred = hsvm_clf.predict(poincare_pts_to_hyperboloid(X_test, eps=1e-6, metric='minkowski'))
    return accuracy_score(test_labels, y_pred)*100, time.time() - start


def euclidean_svm(X_train, train_labels, X_test, test_labels, C, multiclass=False):
    start = time.time()
    n_classes = train_labels.max() + 1
    n_train_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    if multiclass:
        # there is more than 2 classes, using ovr strategy
        # find optimal p for each ovr classifier
        test_probability = np.zeros((n_test_samples, n_classes), dtype=float)
        for class_label in range(n_classes):
            # print('Processing class:', class_label)
            binarized_labels = []
            for i in range(n_train_samples):
                if train_labels[i] == class_label:
                    binarized_labels.append(1)
                else:
                    binarized_labels.append(-1)
            binarized_labels = np.array(binarized_labels)
            linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=True)
            linear_svm.fit(X_train, binarized_labels)
            # print('binary SVM done!')
            w = linear_svm.coef_[0]
            b = linear_svm.intercept_[0]
            decision_vals = np.array([np.dot(w, x) + b for x in X_train])
            ab = SigmoidTrain(deci=decision_vals, label=binarized_labels, prior1=None, prior0=None)
            # print('Platt probability computed!')
            # map testing data using log map
            for i in range(n_test_samples):
                test_decision_val = np.dot(w, X_test[i]) + b
                test_probability[i, class_label] = SigmoidPredict(deci=test_decision_val, AB=ab)
            # print('Probability for each test samples computed!')
        y_pred = np.argmax(test_probability, axis=1)
    else:
        linear_svm = LinearSVC(penalty='l2', loss='squared_hinge', C=C, fit_intercept=True)
        linear_svm.fit(X_train, train_labels)
        y_pred = linear_svm.predict(X_test)
    return accuracy_score(test_labels, y_pred)*100, time.time() - start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments on real data")
    parser.add_argument("--dataset_name", type=str, default='cifar', help="Which dataset to test")
    parser.add_argument("--trails", type=int, default=5, help="How many trails to run")
    parser.add_argument("--refpt", type=str, default='precompute', help="raw or precompute. Reference point")
    parser.add_argument('--save_path', type=str, default="results", help="Where to save results")
    args = parser.parse_args()

    acc = np.zeros((3, args.trails), dtype=float)
    time_used = np.zeros((3, args.trails), dtype=float)
    # load data
    data = np.load('embedding/{}_poincare_embedding.npz'.format(args.dataset_name))
    
    if args.refpt == 'precompute':
#     Use precomputed p
        print('Use precomputed p')
        p = np.load('embedding/{}_reference_points_gt.npy'.format(args.dataset_name))
    elif args.refpt == 'raw':
#     Compute p from scratch
        print('Compute p from scratch')
        if data['x_train'].shape[1]>2:
            print('Convex hull algorithm only support 2d data now! Load precomputed p instead.')
            print('Use precomputed p')
            p = np.load('embedding/{}_reference_points_gt.npy'.format(args.dataset_name))
        else:
            p_list = []
            for label in np.unique(data['y_train']):
                # Find all training points with label
                X = data['x_train'][data['y_train']==label]
                # Find all training points for the rest labels
                X_rest = data['x_train'][data['y_train']!=label]
                # Find convex hull for these two group of points
                CH_label = ConvexHull(X.transpose())
                CH_rest = ConvexHull(X_rest.transpose())
                # Find min dist pair on these two convex hull
                MDP = minDpair(CH_label,CH_rest)
                # choose p as their mid point
                p_list.append(Weightedmidpt(MDP[:,0],MDP[:,1],0.5))

            p = np.stack(p_list, axis=0)
    
    
    
    
    for i in range(args.trails):
        acc[0, i], time_used[0, i] = tangent_hsvm(data['x_train'], data['y_train'].astype(int), data['x_test'],
                                      data['y_test'].astype(int), C=5, p_arr=p, multiclass=True)
        print('Poincare SVM:', acc[0, i], time_used[0, i])

        acc[1, i], time_used[1, i] = cho_hsvm(data['x_train'], data['y_train'].astype(int), data['x_test'],
                                      data['y_test'].astype(int), C=5, multiclass=True)
        print('Cho SVM:', acc[1, i], time_used[1, i])

        acc[2, i], time_used[2, i] = euclidean_svm(data['x_train'], data['y_train'].astype(int), data['x_test'],
                                       data['y_test'].astype(int), C=5, multiclass=True)
        print('Euclidean SVM:', acc[2, i], time_used[2, i])

    print('=' * 10 + 'acc' + '=' * 10)
    print(np.mean(acc, axis=1))
    print(np.std(acc, axis=1))

    print('=' * 10 + 'time_used' + '=' * 10)
    print(np.mean(time_used, axis=1))
    print(np.std(time_used, axis=1))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    np.savez('{}/{}_results.npz'.format(args.save_path, args.dataset_name), acc=acc, time_used=time_used)


