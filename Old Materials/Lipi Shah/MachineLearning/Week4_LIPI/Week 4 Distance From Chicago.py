import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas

DistanceFromChicago = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\DistanceFromChicago.csv',
                      delimiter=',', index_col='CityState')

nCity = DistanceFromChicago.shape[0]

trainData = numpy.reshape(numpy.asarray(DistanceFromChicago['DrivingMilesFromChicago']), (nCity, 1))

KClusters = 15

kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(trainData)

silhouette_avg = metrics.silhouette_score(trainData, kmeans.labels_)

WCSS = numpy.zeros(KClusters)
nC = numpy.zeros(KClusters)

Elbow = 0
for k in range(KClusters):
    count = 0
    sum = 0
    for i in range(nCity):
        if (kmeans.labels_[i] == k):
            count += 1
            diff = trainData[i] - kmeans.cluster_centers_[k]
            sum += diff * diff
    nC[k] = count
    WCSS[k] = sum
    Elbow += sum / count

print("Elbow W = ", Elbow)

for k in range(KClusters):
    print("Cluster ", k)
    print(DistanceFromChicago[kmeans.labels_ == k])

# Display the 4-cluster solution
kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

ClusterResult = DistanceFromChicago
ClusterResult['ClusterLabel'] = kmeans.labels_

for i in range(4):
    print("Cluster Label = ", i)
    print(ClusterResult[kmeans.labels_ == i])