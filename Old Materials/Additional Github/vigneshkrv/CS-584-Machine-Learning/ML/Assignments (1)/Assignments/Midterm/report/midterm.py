# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:17:46 2018

@author: suhas
"""

import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import matplotlib.pyplot as plt
import math
import numpy as np

trainData = pandas.read_csv('ChicagoCompletedPotHole.csv',
                       delimiter=',', usecols=['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION',
'LATITUDE','LONGITUDE'])

trainData['N_POTHOLES_FILLED_ON_BLOCK'] = np.log(trainData['N_POTHOLES_FILLED_ON_BLOCK'])
trainData['N_DAYS_FOR_COMPLETION'] = np.log(1 + trainData['N_DAYS_FOR_COMPLETION'])

nCars = trainData.shape[0]

# Part (a)
nClusters = numpy.zeros(14)
Elbow = numpy.zeros(14)
Silhouette = numpy.zeros(14)
TotalWCSS = numpy.zeros(14)
Inertia = numpy.zeros(14)

KClusters = 1
for c in range(14):
   KClusters = KClusters + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20181010).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCars):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   print("Cluster Assignment:", kmeans.labels_)
   for k in range(KClusters):
      print("Cluster ", k)
      print("Centroid = ", kmeans.cluster_centers_[k])
      print("Size = ", nC[k])
      print("Within Sum of Squares = ", WCSS[k])
      print(" ")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(14):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Part (b)
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.title("Elbow Method")
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.title("Silhouette method")
plt.show()
