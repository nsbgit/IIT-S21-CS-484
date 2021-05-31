

import pandas
import numpy
import sklearn.neighbors

# dataset: CASE_ID,x1,x2,y
X = numpy.matrix([
    [1, 7.7, -37, 4],
    [2, 9.5, -38, 1],
    [3, 3.0, -34, 1],
    [4, 9.1, -75, 1],
    [5, 2.2, -31, 2],
    [6, 4.8, -7, 4],
    [7, 5.5, -6, 3],
    [8, 10, -61, 1],
    [9, 4.2, -23, 2],
    [10, 1.6, -54, 1],
])

for m in ('euclidean', 'manhattan', 'chebyshev'):
    print('Scores for', m)
    for i in range(1, 10):
        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=i, algorithm='brute', metric=m).fit(X)
        distances, indicies = nbrs.kneighbors(X)
        print('\twhen neighbors =', i,
            'distance = ', sum(map(sum,distances)))