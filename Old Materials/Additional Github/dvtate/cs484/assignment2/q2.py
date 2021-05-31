import pandas as pd
import sklearn.cluster as cluster

df = pd.read_csv('cafe.csv')
X = pd.DataFrame({'x' : list(df['Frequency'])})
cluster = cluster.KMeans(n_clusters = 3, random_state = 0).fit(X)

print('Cluster Assignment:', cluster.labels_)
print('Cluster Centroid 0:', cluster.cluster_centers_[0])
print('Cluster Centroid 1:', cluster.cluster_centers_[1])
print('Cluster Centroid 1:', cluster.cluster_centers_[2])
