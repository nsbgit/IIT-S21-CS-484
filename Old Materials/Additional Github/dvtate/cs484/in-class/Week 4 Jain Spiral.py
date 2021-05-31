import math
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy

import numpy.linalg as linalg
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors

Spiral = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\jain.csv',
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
for numberOfNeighbors in numpy.arange(1, 20):
    trainData = Spiral[['x','y']]
    kNNSpec = neighbors.NearestNeighbors(n_neighbors = numberOfNeighbors, algorithm = 'brute', metric = 'euclidean')
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

    # Part (d)
    averageTraceL = numpy.trace(Lmatrix) / nObs
    for i in range(numberOfNeighbors):
        threshEigenvalue = evals[i,] / averageTraceL
        print("i = %2d, Eigenvalue = %.14e %.14e" % (i, evals[i,], threshEigenvalue))

    # Series plot of the smallest five eigenvalues to determine the number of clusters
    sequence = numpy.arange(1,(numberOfNeighbors+1),1) 
    plt.plot(sequence, evals[0:numberOfNeighbors,], marker = "o")
    plt.xlabel('Sequence')
    plt.ylabel('Eigenvalue')
    plt.xticks(sequence)
    plt.grid("both")
    plt.show()

    Z = evecs[:,[0,1]]

    # Perform 2-cluster K-mean on the first two eigenvectors
    kmeans_spectral = cluster.KMeans(n_clusters = 2, random_state = 0).fit(Z)
    Spiral['SpectralCluster'] = kmeans_spectral.labels_

    plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['SpectralCluster'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
