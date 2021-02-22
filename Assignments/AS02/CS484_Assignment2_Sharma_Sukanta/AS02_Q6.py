# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:08:21 2021

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Splring 2021
"""

import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def ManhattanDistance (x, y):
    outDistance = numpy.sum(numpy.abs(x - y))
    return outDistance

df = pd.read_csv('cars.csv' , usecols = ['Weight','Wheelbase','Length'])
scaler = MinMaxScaler(feature_range = (0, 10))
scaler.fit(df)
trainData = scaler.transform(df)



nObs = trainData.shape[0]

# Determine the number of clusters
minNClusters = 2
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
Silhouette = numpy.zeros(maxNClusters)
Calinski_Harabasz = numpy.zeros(maxNClusters)
Davies_Bouldin = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)


for c in range(maxNClusters):
   KClusters = c + 1
   if KClusters < minNClusters:
       continue
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=60616).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (1 < KClusters):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
       Calinski_Harabasz[c] = metrics.calinski_harabasz_score(trainData, kmeans.labels_)
       Davies_Bouldin[c] = metrics.davies_bouldin_score(trainData, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN
       Calinski_Harabasz[c] = numpy.NaN
       Davies_Bouldin[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData[i] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   # if KClusters == 4:
   #     print("SUkanta")
   #     print("Cluster Assignment:", kmeans.labels_)
   #     for k in range(KClusters):
   #        print("Cluster ", k)
   #        print("Centroid = ", kmeans.cluster_centers_[k])
   #        print("Size = ", nC[k])
   #        print("Within Sum of Squares = ", WCSS[k])
   #        print(" ")
    

# print("N Clusters,Inertia,Total WCSS,Elbow Value,Silhouette Value")
# for c in range(minNClusters - 1, maxNClusters):
#    print('{:.0f},{:.4f},{:.4f},{:.4f},{:.4f}'
#          .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))
   
# ----------- Question6.a  ----------------------------------------

print("N Clusters,Elbow values,Silhouette values,Calinski-Harabasz Scores,Davies-Bouldin Indices")
for c in range(minNClusters - 1, maxNClusters):
   print('{:.0f},{:.4f},{:.4f},{:.4f},{:.4f}'
         .format(nClusters[c], Elbow[c], Silhouette[c], Calinski_Harabasz[c], Davies_Bouldin[c]))

# ----------- Question6.b  ----------------------------------------

plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

# ----------- Question6.c  ----------------------------------------
suggestedClusterNo = 4
kmeans4 = cluster.KMeans(n_clusters=suggestedClusterNo, random_state=60616).fit(df)
print()
for k in range(suggestedClusterNo):
   print("\nCluster ", k)
   print("Centroid = ", kmeans4.cluster_centers_[k])
