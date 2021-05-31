import pandas

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import sklearn.cluster as cluster

Spiral = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\jain.csv',
                         delimiter=',')

nObs = Spiral.shape[0]

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['group']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

import sklearn.neighbors

# Three nearest neighbors
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 8, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')

distances = distObject.pairwise(trainData)

import math

Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

# Inspect the values of the selected eigenvectors 
Z = evecs[:,[0,1]]
print(Z[[0]].mean(), Z[[0]].std())
print(Z[[1]].mean(), Z[[1]].std())

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=0).fit(Z)

Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['SpectralCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
