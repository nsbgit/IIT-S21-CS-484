
import pandas
import numpy
import sklearn
import sklearn.neighbors

df = pandas.read_csv('q9.csv')
X = df[['x1', 'x2']]
Y = df['v']

nbrs = sklearn.neighbors.NearestNeighbors(
    n_neighbors=3, algorithm='brute', metric='euclidean').fit(X, Y)

inds = nbrs.kneighbors([[1, 2]], n_neighbors=3, return_distance=False)
print(inds)

for i in inds[0]:
    print(Y[i])