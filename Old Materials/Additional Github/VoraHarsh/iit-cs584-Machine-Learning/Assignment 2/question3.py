import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.neighbors
import math

spiral = pd.read_csv('FourCircle.csv')

nObs = spiral.shape[0]

#Q.1 (5 points) Plot y on the vertical axis versus x on the horizontal axis.  How many clusters are there based on your visual inspection?
print("Question 1 Graph Plotted")
plt.scatter(x=spiral['x'], y=spiral['y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Plot')
plt.grid(True)
plt.show()

#Q2. (5 points) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the scatterplot using the K-mean cluster identifiers to control the color scheme. Please comment on this K-mean result.
print("Question 2: Graph Plotted")
trainData = spiral[['x', 'y']]
kmeans = cluster.KMeans(n_clusters=4, random_state=60616).fit(trainData)
spiral['KMeanCluster'] = kmeans.labels_
for i in range(4):
    spiral.loc[spiral['KMeanCluster'] == i]

plt.scatter(spiral['x'], spiral['y'], c=spiral['KMeanCluster'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-mean Cluster Plot')
plt.grid(True)
plt.show()

#Q3. (10 points) Apply the nearest neighbor algorithm using the Euclidean distance.  We will consider the number of neighbors from 1 to 15.  What is the smallest number of neighbors that we should use to discover the clusters correctly?  Remember that we may need to try a couple of values first and use the eigenvalue plot to validate our choice.
print("Question 3: Graph Plotted")
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i, j] = math.exp(- distances[i][j])
            Adjacency[j, i] = Adjacency[i, j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i, j]
    Degree[i, i] = sum

Lmatrix = Degree - Adjacency

from numpy import linalg as LA

evals, evecs = LA.eigh(Lmatrix)

print("Eigen Values: ")
for i in range(4):
    print("{:e}".format(evals[i]))

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

sequence = np.arange(1,16,1)
plt.plot(sequence, evals[0:15,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.title('Nearest Neighbors')
plt.xticks(sequence)
plt.grid("both")
plt.show()

#Q4. (5 points) Using your choice of the number of neighbors in (c), calculate the Adjacency matrix, the Degree matrix, and finally the Laplacian matrix. How many eigenvalues do you determine are practically zero?  Please display their calculated values in scientific notation.
print("Question 4: Graph Plotted")

Z = evecs[:,[0,1]]

plt.scatter(1e10*Z[:,0], Z[:,1])
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.grid("both")
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=4, random_state=60616).fit(Z)

spiral['SpectralCluster'] = kmeans_spectral.labels_

print("Question 5: Graph Plotted")
plt.scatter(spiral['x'], spiral['y'], c=spiral['SpectralCluster'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-mean cluster identifier')
plt.grid(True)
plt.show()
