#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, precision_score, auc
import sys
import math

from utils.utils import *
from utils.datasets import *
from platt import *

# -----------------------------------------------
# ----- Hyperbolic Support Vector Classifier ----
# -----------------------------------------------


def loss_fn(w, X, y, C=1.0):
    """
    Compute loss function for HSVM maximum margin formulation
    """
    margin_loss = -1/2 * minkowski_dot(w, w)
    misclass_loss = np.arcsinh(1) - np.arcsinh(y * [minkowski_dot(w, x) for x in X])
    return margin_loss + C*np.sum(np.where(misclass_loss > 0, misclass_loss, 0))


def grad_fn(w, X, y, C=1.0, eps=1e-5):
    """
    Solve for gradient w.r.t w for loss function L(w, x, y)
    """
    # if len(y.shape) < 2:
    #    y = y.reshape(-1, 1)
    grad_margin = np.hstack((-w[0], w[1:]))
    z = y*np.array([minkowski_dot(w, x) for x in X])
    misclass = (np.arcsinh(1) - np.arcsinh(z)) > 0
    arcsinh_term = -1/np.sqrt(z**2 + 1)
    mink_prod_term = y.reshape(-1,1)*np.hstack((X[:, 0].reshape(-1, 1), -1*X[:, 1:]))
    grad_misclass = misclass.reshape(-1,1) * arcsinh_term.reshape(-1,1) * mink_prod_term
    grad_w = grad_margin + C*np.sum(grad_misclass, axis=0)
    return grad_w


def is_feasible(w):
    """
    Check if weight vector is in feasible region
    """
    norm_cond = minkowski_dot(w, w) < 0
    return norm_cond


def proj_boundary(w, alpha, eps=1e-6):
    """
    Define distance to projection as a function of alpha
    """
    proj_w = w.copy()
    proj_w[1:] = (1 + alpha)*proj_w[1:]
    first_sgn = 1 if proj_w[0] >= 0 else -1
    proj_w[0] = first_sgn * np.sqrt(np.sum(proj_w[1:]**2) - eps)
    return proj_w


def alpha_search(w):
    """
    Use scipy to solve for projection by minimizing distance
    """
    res = sp.optimize.minimize_scalar(lambda alpha: np.sum((proj_boundary(w, alpha) - w)**2))
    alpha = res.x
    return alpha


def train_hsvm(w, X, y, C, num_epochs, lr, batch_size, max_lr_attempts, early_stopping, verbose):
    num_samples = len(y)
    y = 2*y - 1
    early_stop_count = 0
    lr_attempts = 0
    init_w = w
    best_w = init_w
    if not is_feasible(init_w):
        init_w = proj_boundary(init_w, alpha=0.01)
    if is_feasible(init_w):
        w_new = init_w
        init_loss = loss_fn(init_w, X, y, C)
        min_loss = init_loss
        for j in range(num_epochs):
            current_loss = 0
            shuffle_index = np.random.permutation(num_samples)
            for i in range(0, math.ceil(num_samples / batch_size)):
                Xj = X[shuffle_index[i*batch_size:max((i+1)*batch_size, num_samples)]]
                yj = y[shuffle_index[i*batch_size:max((i+1)*batch_size, num_samples)]]
                grad_w = grad_fn(w_new, Xj, yj, C)
                w_new = w_new - lr*grad_w
                # if not in feasible region, need to use projection
                if not is_feasible(w_new):
                    # solve optimization problem for nearest feasible point
                    alpha_opt = alpha_search(w_new)
                    # project w to feasible sub-space
                    w_new = proj_boundary(w_new, alpha_opt)
                current_loss += loss_fn(w_new, Xj, yj, C)
            current_loss = current_loss / num_samples
            if current_loss < min_loss:
                min_loss = current_loss
                best_w = w_new
            else:
                early_stop_count += 1
                if early_stop_count >= early_stopping:
                    if lr_attempts < max_lr_attempts:
                        lr_attempts += 1
                        lr = lr / 2
                        early_stop_count = 0
                        continue
                    else:
                        if verbose:
                            print('Exiting early: no decrease for ' + str(early_stopping) + ' rounds')
                        break
            if verbose:
                print('COMPLETED EPOCH ', j+1)
                print('-------- LOSS: ', current_loss)
    return best_w


class LinearHSVM():
    """
    Hyperbolic support vector classification model trained in the hyperboloid model 
    with an iterative, projected gradient descent method
    
    """
    
    def __init__(self, C=1.0, num_epochs=20, lr=0.001, tol=1e-8, batch_size=20, 
                 early_stopping=5, max_retries=3, verbose=False, multiclass = False):
        self.C = C
        self.num_epochs = num_epochs
        self.tol = tol
        self.verbose = verbose
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.lr = lr
        self.max_retries = max_retries
        self.multiclass = multiclass
        
    def init_weights(self, X, y):
        """
        Initialize w (weights) with the coefficients found from applying
        standard LinearSVC in the ambient Euclidean space
        
        """
        # fit linear svm using scikit-learn classifier
        linear_svm = LinearSVC(fit_intercept=False, loss='hinge', C=self.C)
        self.init_coef = linear_svm.fit(X, y).coef_[0]
    
    def fit(self, X, y):
        """
        Train linear HSVM model for input data X, labels y
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples, 1)
        """
        # set attribute for training labels
        self.y_train = y
        if self.multiclass:
            y_binary = label_binarize(y, classes=np.unique(y))
            self.class_labels_ = np.unique(y)
            self.coef_ = []
            self.platt_coefs_ = []
            for i in range(len(np.unique(y))):
                self.init_weights(X, y_binary[:, i])
                wi = train_hsvm(self.init_coef, X, y_binary[:, i], self.C, 
                                        self.num_epochs, self.lr, self.batch_size,
                                        early_stopping=self.early_stopping,
                                        max_lr_attempts=self.max_retries, verbose=self.verbose)
                self.coef_.append(wi)
                
            # get platt coefficients for probability scaling
            for i in range(len(np.unique(y))):
                decision_vals = np.array([minkowski_dot(self.coef_[i], x) for x in X])
                # get binary labels {0, 1} for the separate 'OVR' classifiers
                yi_train = (self.y_train == self.class_labels_[i]).astype('int')
                # convert labels = {0, 1} to {-1, 1}
                yi_train = 2*yi_train - 1
                # get platt coefs A, B
                ab = SigmoidTrain(deci=decision_vals, label=yi_train, prior1=None, prior0=None)
                self.platt_coefs_.append(ab)
        else:    
            self.init_weights(X, y)
            classes = np.unique(y)
            self.class_labels_ = {'neg_class': np.min(classes), 'pos_class': np.max(classes)}
            self.coef_ = train_hsvm(self.init_coef, X, y, self.C, 
                                    self.num_epochs, self.lr, self.batch_size,
                                    early_stopping=self.early_stopping,
                                    max_lr_attempts=self.max_retries, verbose=self.verbose)
            decision_vals = np.array([minkowski_dot(self.coef_, x) for x in X])
            # convert labels = {0, 1} to {-1, 1}
            yi_train = 2*self.y_train - 1
            # get platt coefs A, B
            ab = SigmoidTrain(deci=decision_vals, label=yi_train, prior1=None, prior0=None)
            self.platt_coefs_ = ab
        return self

    def predict(self, X):
        """
        Predict class labels with hyperbolic linear decision function
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        """
       
        if self.multiclass:
            n_classes = len(self.coef_)
            
            # we find probabilities of belonging to each class 
            y_probs = np.zeros((X.shape[0], n_classes))
            
            # find each class prediction score and apply Platt probability scaling
            for i in range(n_classes):
                decision_vals = np.array([minkowski_dot(self.coef_[i], x) for x in X])
                for j in range(X.shape[0]):
                    y_probs[j, i] = SigmoidPredict(deci=decision_vals[j], AB=self.platt_coefs_[i])
            # set prediction label to the highest probability class
            y_pred = self.class_labels_[np.argmax(y_probs, axis=1)]
        else:    
            y_pred = np.zeros((X.shape[0], ))
            vals = np.array([minkowski_dot(self.coef_, x) for x in X])
            y_pred[vals < 0] = self.class_labels_['neg_class']
            y_pred[vals >= 0] = self.class_labels_['pos_class']
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict probability from Platt method and hyperbolic decision function vals
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        """
       
        if self.multiclass:
            n_classes = len(self.coef_)
            
            # we find probabilities of belonging to each class 
            y_probs = np.zeros((X.shape[0], n_classes))
            
            # find each class prediction score and apply Platt probability scaling
            for i in range(n_classes):
                decision_vals = np.array([minkowski_dot(self.coef_[i], x) for x in X])
                for j in range(X.shape[0]):
                    y_probs[j, i] = SigmoidPredict(deci=decision_vals[j], AB=self.platt_coefs_)
            return y_probs
        
        else:
            y_probs = np.zeros((X.shape[0], ))
            decision_vals = np.array([minkowski_dot(self.coef_, x) for x in X])
            for i in range(X.shape[0]):
                y_probs[i] = SigmoidPredict(deci=decision_vals[i], AB=self.platt_coefs_)
            return y_probs
            
    def score(self, X, y):
        """
        Return accuracy evaluated on X, y
        """
        y_pred = self.predict(X)
        return np.sum(y_pred == y) / len(y)
    
    def decision_function(self, X):
        """
        Evaluate minkowski inner product between weights and input data
        """
        if self.multiclass:
            n_classes = len(self.coef_)
            pred_vals = np.zeros((X.shape[0], n_classes))
            for i in range(n_classes):
                pred_vals[:, i] = np.array([minkowski_dot(self.coef_[i], x) for x in X])
        else:    
            pred_vals = np.array([minkowski_dot(self.coef_, x) for x in X])
        return pred_vals
    

# convert array or list [a, b] to complex number a+bi
def to_complex(xi):
    return np.complex(xi[0], xi[1])


# parametrized hyperbolic line with distance s from point z0 in direction gamma
def hyp_line(s, gamma, z0):
    if not isinstance(z0,complex):
        z0 = to_complex(z0)
    if not isinstance(gamma, complex):
        gamma = to_complex(gamma)
    return (gamma * np.tanh(s/2) + z0) / (np.conj(z0) * gamma * np.tanh(s/2) + 1)


# plot hyperbolic line (circular arc) in poincare plane
def plot_hyp_line(dist, gamma, z0, color='black', whole_line=False, z0_label=None, endpts=False):
    ax = plt.gca()
    if np.all(gamma == 0):
        ax.scatter(np.real(z0), np.imag(z0), s=40, alpha=1, c='black')
        ax.text(np.real(z0)-0.05, np.imag(z0)+0.05, z0_label, size=16)
        return
    gamma = gamma / norm(gamma)
    pts = []
    for r in np.arange(0, dist, 0.05):
        zj = hyp_line(r, gamma, z0)
        pts.append(zj)
        if whole_line:
            zj_flip = hyp_line(r, -1*gamma, z0)
            pts.append(zj_flip)
    pts = np.array(pts)
    ax.scatter(np.real(pts), np.imag(pts), s=25, alpha=1, c=color)
    ax.scatter(np.real(z0), np.imag(z0), s=25, alpha=1, c=color)
