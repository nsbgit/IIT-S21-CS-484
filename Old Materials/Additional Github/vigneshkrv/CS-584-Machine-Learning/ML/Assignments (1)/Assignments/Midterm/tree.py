# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:46:04 2018

@author: suhas
"""

import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import matplotlib.pyplot as plt
import math
import numpy as np

orig = pandas.read_csv('ChicagoCompletedPotHole.csv',
                       delimiter=',', usecols=['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE'])


trainData = pandas.read_csv('ChicagoCompletedPotHole.csv',
                       delimiter=',', usecols=['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE'])

trainData['N_POTHOLES_FILLED_ON_BLOCK'] = np.log(trainData['N_POTHOLES_FILLED_ON_BLOCK'])
trainData['N_DAYS_FOR_COMPLETION'] = np.log(1 + trainData['N_DAYS_FOR_COMPLETION'])

kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

trainData['KMeanCluster'] = kmeans.labels_

for i in range(4):
    print("Cluster Label = ", i)
    print(trainData.loc[trainData['KMeanCluster'] == i])

plt.scatter(trainData[['LONGITUDE']], trainData[['LATITUDE']], s= 0.1, c = trainData[['KMeanCluster']])
plt.xlabel('LONGITUDE')
plt.ylabel('LATITUDE')
plt.grid(True)

plt.axes().set_aspect('equal', 'datalim')

plt.show()




from sklearn import tree
from sklearn.metrics import zero_one_loss
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20181010)

hmeq_dt = classTree.fit(orig[['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE']], trainData['KMeanCluster'])
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(classTree.score(orig[['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE']], trainData['KMeanCluster'])))
    
preds = hmeq_dt.predict(orig[['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE']])

print("Missclassifiction rate",zero_one_loss(preds,trainData['KMeanCluster']))

import graphviz
#dot_data = tree.export_graphviz(hmeq_dt, out_file=None)

dot_data = tree.export_graphviz(hmeq_dt,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE'],
                                class_names = ['0', '1','2','3'])

graph = graphviz.Source(dot_data)
graph.render('hmeq_output')

