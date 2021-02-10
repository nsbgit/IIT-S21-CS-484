# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:12:03 2019

@author: Programmer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA

fraud = pd.read_csv('D:\\IIT Edu\\Sem1\\MachineLearning\\Week2\\Fraud.csv',
                       delimiter=',')

#print(fraud)
################################# 3 (a) ########################################
print(np.around(fraud['FRAUD'].value_counts()[1]/fraud.shape[0], 6) * 100) #percentage

################################# 3 (b) ########################################
fradulent = fraud[fraud['FRAUD'] == 1]  
nonFradulent = fraud[fraud['FRAUD'] != 1]
var_Array = fraud.keys().tolist()
var_Array.remove('CASE_ID')
var_Array.remove('FRAUD')

for i in range(0, len(var_Array)):
  boxPlotData =[nonFradulent[var_Array[i]], fradulent[var_Array[i]], ];
  plt.boxplot(boxPlotData, vert=0, labels=[0,1], patch_artist= True);
  plt.title(var_Array[i])
  plt.show();

################################# 3 (c) ########################################
x = np.matrix(fraud.drop(['CASE_ID','FRAUD'], axis=1))
xtx = x.transpose() * x
evalues, evects = LA.eigh(xtx)

#print(evalues)
#print(evects)

transf = evects * LA.inv(np.sqrt(np.diagflat(evalues)));
print("Transformation Matrix = \n", transf)

transf_x = x * transf;
print("The Transformed x = \n", transf_x)
xtx = transf_x.transpose() * transf_x;
print("Identity Matrix = \n", xtx)
xtx.shape

from scipy import linalg as LA2

orthx = LA2.orth(x)
print("The orthonormalize x = \n", orthx)


check = orthx.transpose().dot(orthx)
print("Identity Matrix = \n", check)

################################# 3 (d) ########################################
from sklearn.neighbors import KNeighborsClassifier

trainData = fraud.drop(['CASE_ID','FRAUD'], axis=1)
target = fraud['FRAUD'];

neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)
accu = nbrs.score(x, target)
print(accu)

################################# 3 (e) ########################################
from sklearn.neighbors import NearestNeighbors as knn
focal = [7500,15,3,127,2,2]
transf_focal = focal * transf;
neigh_t = knn(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs_t = neigh_t.fit(transf_x)

myNeighbors_t = nbrs_t.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = ", myNeighbors_t)


############################### 3 (f)#########################################

print("Predicted Probability of Fraudulent ",nbrs.predict(transf_x))






