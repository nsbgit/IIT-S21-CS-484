# Load the PANDAS library
import pandas
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

data = ['A','B','C','D','E','F','G']
count = 0

res1 = combinations(data, 1)
for i in list(res1):
    count = count + 1
    print(i)

res1 = combinations(data, 2)
for i in list(res1):
    count = count + 1
    print(i)
    
res1 = combinations(data, 3)
for i in list(res1):
    count = count + 1
    print(i)
    
res1 = combinations(data, 4)
for i in list(res1):
    count = count + 1
    print(i)
    
res1 = combinations(data, 5)
for i in list(res1):
    count = count + 1
    print(i)
    
res1 = combinations(data, 6)
for i in list(res1):
    count = count + 1
    print(i)
    
res1 = combinations(data, 7)
for i in list(res1):
    count = count + 1
    print(i)
GrocData = pandas.read_csv('Groceries.csv',
                       delimiter=',')

# Examine a portion of the data frame
print(GrocData)

# Create frequency 
nItemPerCustomer = GrocData.groupby(['Customer'])['Item'].count()
freqTable = pandas.value_counts(nItemPerCustomer).reset_index()
freqTable.columns = ['Item', 'Frequency']
freqTable = freqTable.sort_values(by = ['Item'])
print(freqTable)
nItemPerCustomer.describe()

plt.hist(nItemPerCustomer)
plt.xlabel('Number of Unique Items')
plt.ylabel('Frequency')
plt.title('Histogram of the number of unique items')
plt.show()


g0_25, g0_50, g0_75 = np.percentile(nItemPerCustomer, [25, 50, 75])

# Convert the Sale Receipt data to the Item List format
ListItem = GrocData.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.0076258261311, max_len = 32, use_colnames = True)

# Discover the association rules
assoc_rules1 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)

plt.scatter(assoc_rules1['confidence'],assoc_rules1['support'],s = assoc_rules1['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.title("Support vs Confidence w/ Lift metrics indicating size of marker")
plt.show()

assoc_rules60 = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)


import sklearn.cluster as cluster
from sklearn import metrics
from scipy.spatial.distance import cdist

HPnWeight = pandas.read_csv('cars.csv', delimiter=',')

x1 =  HPnWeight['Horsepower']
x2 =  HPnWeight['Weight']
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distortions = []
K = range(1,16)
for k in K:
    kmeanModel = cluster.KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Elbow W')
plt.title('The Elbow Method showing the optimal k')
plt.show()

K = range(2,16)
score = []
for k in K:
    kmeanModel = cluster.KMeans(n_clusters=k).fit(X)
    preds = kmeanModel.fit_predict(X)
    score.append(metrics.silhouette_score(X, preds, metric='euclidean'))
    
plt.plot(K, score, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('The Silhouette Method showing the optimal k')
plt.show()



Spiral = pandas.read_csv('Spiral.csv', delimiter=',')
nObs = Spiral.shape[0]

plt.scatter(Spiral[['x']], Spiral[['y']])
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
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
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