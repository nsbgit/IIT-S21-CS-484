# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import sklearn.cluster as cluster

X = np.array([[0.1], [0.3], [0.4], [0.8], [0.9]])

kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(X)
print("Cluster Assignment:", kmeans.labels_)
print("Cluster Centroid 0:", kmeans.cluster_centers_[0])
print("Cluster Centroid 1:", kmeans.cluster_centers_[1])

import pandas

DistanceFromChicago = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\DistanceFromChicago.csv',
                      delimiter=',', index_col='CityState')

nCity = DistanceFromChicago.shape[0]

D = np.reshape(np.asarray(DistanceFromChicago['DrivingMilesFromChicago']), (nCity, 1))

kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(D)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

ClusterResult = DistanceFromChicago
ClusterResult['ClusterLabel'] = kmeans.labels_

for i in range(4):
    print("Cluster Label = ", i)
    print(ClusterResult.loc[ClusterResult['ClusterLabel'] == i])
    
    