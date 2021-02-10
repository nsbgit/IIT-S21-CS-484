#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import matplotlib.pyplot as plt
import math


# In[2]:


data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4\\ChicagoCompletedPotHole.csv',
                  delimiter=',' , usecols=['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE'])


# In[3]:


data['N_POTHOLES_FILLED_ON_BLOCK'] = np.log(data['N_POTHOLES_FILLED_ON_BLOCK'])
data['N_DAYS_FOR_COMPLETION'] = np.log(1 + data['N_DAYS_FOR_COMPLETION'])


# In[4]:


nClusters = np.zeros(9)
Elbow = np.zeros(9)
Silhouette = np.zeros(9)
TotalWCSS = np.zeros(9)
Inertia = np.zeros(9)

nRows = data.shape[0]

for c in range(9):
    KClusters = c + 2
    nClusters[c] = KClusters

    kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20200304).fit(data)

    # The Inertia value is the within cluster sum of squares deviation from the centroid
    Inertia[c] = kmeans.inertia_
   
    if (1 < KClusters & KClusters < 11):
        Silhouette[c] = metrics.silhouette_score(data, kmeans.labels_)
    else:
        Silhouette[c] = np.NaN

    WCSS = np.zeros(KClusters)
    nC = np.zeros(KClusters)

    for i in range(nRows):
        k = kmeans.labels_[i]
        nC[k] += 1
        diff = data.iloc[i,] - kmeans.cluster_centers_[k]
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


# In[5]:


print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(9):
    print('{:.0f} \t\t {:.7f} \t {:.7f} \t {:.7f} \t {:.7f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))


# In[6]:


plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(np.arange(2, 11, step = 1))
plt.show()


# In[7]:


plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(np.arange(2, 11, step = 1))
plt.show()

