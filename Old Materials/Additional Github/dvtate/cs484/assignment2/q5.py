# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.cluster

# Load data
df = pd.read_csv('FourCircle.csv')

###
# Q5.a
###
print('q5a')

x = df['x']
y = df['y']

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

###
# Q5.b
###
print('q5b')

kmcs = sklearn.cluster.KMeans(n_clusters=4, ).fit(df)
color_legend = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
colors = list(map(lambda i: color_legend[i], kmcs.labels_))

plt.scatter(x, y, color=colors)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

###
# Q5.c
###
print('q5c')

from sklearn.neighbors import KNeighborsClassifier

def check_accuracy(n_neighbors):
    # Perform classification
    train_data = df[['x', 'y']]
    target = df['ring']
    neigh = KNeighborsClassifier(
        n_neighbors = n_neighbors,
        algorithm = 'brute',
        metric = 'euclidean'
    ).fit(train_data, target)

    return neigh.score(train_data, target)

accuracies = list(map(check_accuracy, range(1, 16)))
print(accuracies)

###
# Q5.de
###
print('q5e-d')

# Math go brrr
sccs = sklearn.cluster.SpectralClustering(
    n_neighbors = 1,
    n_clusters  = 4,
    n_components= 3,
).fit(df)
color_legend = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
colors = list(map(lambda i: color_legend[i], sccs.labels_))

plt.scatter(x, y, color=colors)
plt.xlabel('x')
plt.ylabel('y')
plt.show()