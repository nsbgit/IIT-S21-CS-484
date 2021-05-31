#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import iqr
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from numpy import linalg as LA
import sklearn.neighbors
import sklearn.cluster as cluster
from sklearn import metrics
from scipy.spatial.distance import cdist

#---------------------- -------------Q1-A-------------------------------------- 

dataset = pd.read_csv('Groceries.csv')
newData = dict(Counter(dataset['Customer']))
newDataset = pd.DataFrame.from_dict(newData,orient='index',columns=['#items'])
plt.hist(newDataset['#items'])
plt.grid(True)
plt.xlabel('Number of Unique Items')
plt.ylabel('Frequency')
plt.show()
quartiles = np.percentile(newDataset['#items'],[25,50,75])
print('25th, Median, 75th percentiles are',quartiles)


#-------------------------------------- Q1-B-------------------------------------------------
ListOfItems = dataset.groupby(['Customer'])['Item'].apply(list).values.tolist()
trans_encoding = TransactionEncoder()
trans_encoding_array = trans_encoding.fit(ListOfItems).transform(ListOfItems)
transaction_tab = pd.DataFrame(trans_encoding_array, columns=trans_encoding.columns_)
transaction_tab



itemSetsFrequency = apriori(transaction_tab, min_support = 0.0076258261311, max_len = 32, use_colnames = True)
itemSetsFrequency

#------------------------------------- Q1-C--------------------------------------------------
assocRule = association_rules(itemSetsFrequency, metric = "confidence", min_threshold = 0.01)
print(len(assocRule), 'associations rules where confidence metric is atleast 1% ')

#----------------------------------------- Q1-D---------------------------------------------------
plt.scatter(assocRule['confidence'],assocRule['support'],s = assocRule['lift'])
plt.grid(True)
plt.xlabel("conf")
plt.ylabel("supp")
plt.title("support vs confidence")
plt.show()


#----------------------------------------- Q1-E---------------------------------------------------
assocRule = association_rules(itemSetsFrequency, metric = "confidence", min_threshold = 0.6)
assocRule
# print('All consequents are whole milk')


spiralSet = pd.read_csv('Spiral.csv')
spiralSet[:10]

#-------------------------------------------- Q2-A------------------------------------------------
plt.scatter(spiralSet['x'],spiralSet['y'])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clusters')
plt.show()

#------------------------------------------- Q2-B----------------------------------------------------
trainSet = spiralSet[['x','y']]
kmeans = cluster.KMeans(n_clusters=2,random_state=60616).fit(trainSet)
print("Cluster Centroids = \n", kmeans.cluster_centers_)

spiralSet['KMeanCluster'] = kmeans.labels_

color = []
for i in range(len(spiralSet)):
    if spiralSet['KMeanCluster'][i] == 0:
        color.append('green')
    elif spiralSet['KMeanCluster'][i] ==1:
        color.append('orange')
        
for i in range(2):
    print("\nCluster Label = ", i)
    print(spiralSet.loc[spiralSet['KMeanCluster'] == i])

plt.scatter(spiralSet[['x']], spiralSet[['y']],color=color)
plt.title('Clusters')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


#-------------------------------------------------- Q2-C--------------------------------------------

kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainSet)
dist, indices = nbrs.kneighbors(trainSet)
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
dists = distObject.pairwise(trainSet)

adjacent = np.zeros((spiralSet.shape[0], spiralSet.shape[0]))
degree = np.zeros((spiralSet.shape[0], spiralSet.shape[0]))

for i in range(spiralSet.shape[0]):
    for j in indices[i]:
        if (i <= j):
            adjacent[i,j] = math.exp(- dists[i][j])
            adjacent[j,i] = adjacent[i,j]

for i in range(spiralSet.shape[0]):
    sum = 0
    for j in range(spiralSet.shape[0]):
        sum += adjacent[i,j]
    degree[i,i] = sum
        
Lmatrix = degree - adjacent

#------------------------------------ Q2-D--------------------------------------
eigen_values, eigen_vectors = LA.eigh(Lmatrix)
E1 = eigen_vectors[:,[0,1]]

print('\nRounding upto 10 decimal places mean and std\n')
print(np.round(E1[[0]].mean(),10), np.round(E1[[0]].std(),10))
print(np.round(E1[[1]].mean(),10), np.round(E1[[1]].std(),10))

plt.title('Graph of first two Eigen Vectors')
plt.scatter(E1[[0]], E1[[1]])
plt.xlabel('first eigen vector')
plt.ylabel('second eigen vector')
plt.show()

# -----------------------------------Q2-E----------------------------------------
kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(E1)
spiralSet['SpectralCluster'] = kmeans_spectral.labels_

color = []
for i in range(len(spiralSet)):
    if spiralSet['SpectralCluster'][i] == 0:
        color.append('yellow')
    elif spiralSet['SpectralCluster'][i] ==1:
        color.append('green')

plt.scatter(spiralSet['x'], spiralSet['y'], c = color)
plt.title('Clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




