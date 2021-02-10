import math
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy

import numpy.linalg as linalg
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors

Spiral = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4_LIPI\\jain.csv',
                         delimiter=',')

nObs = Spiral.shape[0]

plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['group'])
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

plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Fourteen nearest neighbors
kNNSpec = neighbors.NearestNeighbors(n_neighbors = 14, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency matrix
Adjacency = numpy.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())

# Create the Degree matrix
Degree = numpy.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency

# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)

# Series plot of the smallest five eigenvalues to determine the number of clusters
sequence = numpy.arange(1,5,1) 
plt.plot(sequence, evals[0:4,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.xticks(sequence)
plt.grid("both")
plt.show()

# Series plot of the smallest twenty eigenvalues to determine the number of neighbors
sequence = numpy.arange(1,21,1) 
plt.plot(sequence, evals[0:20,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()

# Inspect the values of the selected eigenvectors 
for j in range(10):
    print('Eigenvalue: ', j)
    print('              Mean = ', numpy.mean(evecs[:,j]))
    print('Standard Deviation = ', numpy.std(evecs[:,j]))
    print('  Coeff. Variation = ', scipy.stats.variation(evecs[:,j]))

Z = evecs[:,[0,1]]

plt.scatter(1e10*Z[:,0], Z[:,1])
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.grid("both")
plt.show()

# Perform 2-cluster K-mean on the first two eigenvectors
kmeans_spectral = cluster.KMeans(n_clusters = 2, random_state = 0).fit(Z)
Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
