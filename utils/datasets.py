#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import pandas as pd
import numpy as np
import scipy.io as sio
from .utils import *


# read in matlab data files from: https://github.com/hhcho/hyplinear
def load_mat_file(path):
    # load .mat files with gaussian datasets
    data = sio.loadmat(path)
    X, y = data['B'], data['label'].ravel().astype(np.int)
    # if not in poincare disk, project points within the disk
    if (norm(X, axis=1) > 1).any():
        out_pts = norm(X, axis=1) > 1
        num_pts = np.sum(out_pts)
        X[out_pts] = X[out_pts] / norm(X[out_pts], axis=0) - np.repeat(1e-3, num_pts).reshape(-1,1)
    return X, y