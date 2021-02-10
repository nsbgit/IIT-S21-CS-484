# -*- coding: utf-8 -*-
"""

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Splring 2021
"""

# Load the necessary libraries
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
import scipy
from numpy import linalg as LA
from scipy import linalg as LA2
from sklearn.neighbors import KNeighborsClassifier

# **************************  Question 3  ******************************************
data = pd.read_csv('Fraud.csv', index_col='CASE_ID')

# **************************  Question 3.a  ******************************************
fraudCount = data[data['FRAUD'] == 1].count()['FRAUD']
totalCount = data['FRAUD'].count()
fraudPercentage = (fraudCount / totalCount) * 100
fraudPercentage = float("{0:.4f}".format(fraudPercentage))
print("\n\n\nQ3.a)	(5 points) What percent of investigations are found to be frauds?  Please give your answer up to 4 decimal places.\n")
print("\n{0}%\n".format(fraudPercentage))

# **************************  Question 3.b  ******************************************
x = np.matrix(data)
xtx = x.transpose() * x


evals, evecs = LA.eigh(xtx)
# Want eigenvalues greater than one
evals_1 = evals[evals > 1.0]
evecs_1 = evecs[:,evals > 1.0]
print("Eigenvalues of x = \n", evals_1)
print("Eigenvectors of x = \n",evecs_1)
print("Number of Dimensions = ", x.ndim)


transf = evecs_1 * LA.inv(np.sqrt(np.diagflat(evals_1)));
print("Transformation Matrix = \n", transf)
print()

transf_x = x * transf;
print("The Transformed x = \n", transf_x)

xtx = transf_x.transpose() * transf_x
print("Expecting an Identity Matrix, so the resulting variables are orthonormal= \n", xtx)
print()


# **************************  Question 3.c  ******************************************

# Perform classification
# Specify target: 0 = Asia, 1 = Europe, 2 = USA
# target = cars_wIndex['Origin']

# neigh = KNeighborsClassifier(n_neighbors=4 , algorithm = 'brute', metric = 'euclidean')
# nbrs = neigh.fit(trainData, target)



target = data.iloc[:,0]
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(transf_x, target)
scoreResult = nbrs.score(transf_x, target)
print("\n\nScore function values:{0}\n".format(scoreResult))

# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************