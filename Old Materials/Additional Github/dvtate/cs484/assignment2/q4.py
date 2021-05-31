import pandas
import sklearn.cluster
import sklearn.metrics

print('Loading clusters...')
X = pandas.DataFrame({ 'x' : [-2, -1, 1, 2, 3, 4, 5, 7, 8] })
cs = sklearn.cluster.KMeans(n_clusters = 2, random_state = 0).fit(X)

# Verfiy cluster assignments are correct
print('\tCluster Assignment:', cs.labels_)

###
# Q4.a
###
print('Q4.a')

# Compute sillouette width
sil = sklearn.metrics.silhouette_score(X, cs.labels_)
print('\t', sil)


###
# Q4.c
###
print('Q4.c')

# Compute davies
dbs = sklearn.metrics.davies_bouldin_score(X, cs.labels_)
print('\t', dbs)