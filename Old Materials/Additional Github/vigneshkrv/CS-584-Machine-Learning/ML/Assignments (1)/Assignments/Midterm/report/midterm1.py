# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 16:25:17 2018

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
orig['KMeanCluster'] = kmeans.labels_

for i in range(4):
    print("Cluster Label = ", i)
    print(trainData.loc[trainData['KMeanCluster'] == i])
    
#box_npot = orig.boxplot(column='N_POTHOLES_FILLED_ON_BLOCK', by='KMeanCluster', figsize=(6,4))
#box_npot.set_ylabel('N_POTHOLES_FILLED_ON_BLOCK')
#box_npot.set_title("")
#
#box_ndays = orig.boxplot(column='N_DAYS_FOR_COMPLETION', by='KMeanCluster', figsize=(6,4))
#box_ndays.set_ylabel('N_DAYS_FOR_COMPLETION')
#box_ndays.set_title("")
#
#box_lat = orig.boxplot(column='LATITUDE', by='KMeanCluster', figsize=(6,4))
#box_lat.set_ylabel('LATITUDE')
#box_lat.set_title("")
#
#box_long = orig.boxplot(column='LONGITUDE', by='KMeanCluster', figsize=(6,4))
#box_long.set_ylabel('LONGITUDE')
#box_long.set_title("")

plt.scatter(trainData[['LONGITUDE']], trainData[['LATITUDE']], s= 0.1, c = trainData[['KMeanCluster']])
plt.xlabel('LONGITUDE')
plt.ylabel('LATITUDE')
plt.title("Scatterplot of LATITUDE versus LONGITUDE w/ Cluster ID as color response")
plt.grid(True)

plt.axes().set_aspect('equal', 'datalim')

plt.show()