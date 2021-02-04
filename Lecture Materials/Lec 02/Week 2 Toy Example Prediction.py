# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:36:25 2021

@author: pc
"""

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsRegressor

toy_example = pandas.read_csv("Week 2 Toy Example.csv", header = 0)

# Specify the data
X = toy_example[['x1', 'x2']]
Y = toy_example['y']

# Build nearest neighbors
kNNSpec = KNeighborsRegressor(n_neighbors=2, metric='euclidean')
nbrs = kNNSpec.fit(X, Y)
distances, indices = nbrs.kneighbors(X)

# Calculate prediction, errors, and sum of squared error
pred_y = nbrs.predict(X)
error_y = Y - pred_y
sse_y = np.sum(error_y ** 2)
