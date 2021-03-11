import matplotlib.pyplot as plt
import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas as pd

potHoleData = pd.read_csv('ChicagoCompletedPotHole.csv', usecols = ['N_POTHOLES_FILLED_ON_BLOCK','N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE'])

nRow = potHoleData.shape[0]
nCol = potHoleData.shape[1]

# trainData = numpy.reshape(numpy.asarray(potHoleData['DrivingMilesFromChicago']), (nCity, 1))
trainData = numpy.reshape(numpy.asarray(potHoleData), (nRow, nCol))
# trainData = trainData[0: 20 ,  :]
data1 = trainData[:, 0]
data2 = trainData[:, 1]
data3 = trainData[:, 2]
data4 = trainData[:, 3]
data1 = numpy.log(data1)
data2 = data2 + 1
data2 = numpy.log(data2)
trainData = numpy.column_stack((data1, data2, data3, data4))


nObs = trainData.shape[0]

# Determine the number of clusters
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters)
Elbow = numpy.zeros(maxNClusters)
Silhouette = numpy.zeros(maxNClusters)
Calinski_Harabasz = numpy.zeros(maxNClusters)
Davies_Bouldin = numpy.zeros(maxNClusters)
TotalWCSS = numpy.zeros(maxNClusters)
Inertia = numpy.zeros(maxNClusters)

for c in range(maxNClusters):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20201014).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (1 < KClusters):# and (KClusters < len(kmeans.labels_))):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)
       Calinski_Harabasz[c] = metrics.calinski_harabasz_score(trainData, kmeans.labels_)
       Davies_Bouldin[c] = metrics.davies_bouldin_score(trainData, kmeans.labels_)
   else:
       Silhouette[c] = numpy.NaN
       Calinski_Harabasz[c] = numpy.NaN
       Davies_Bouldin[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData[i] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   # print("Cluster Assignment:", kmeans.labels_)
   # for k in range(KClusters):
   #    print("Cluster ", k)
   #    print("Centroid = ", kmeans.cluster_centers_[k])
   #    print("Size = ", nC[k])
   #    print("Within Sum of Squares = ", WCSS[k])
   #    print(" ")

print("N Clusters,Inertia,Total WCSS,Elbow Value,Silhouette Value,Davies-Bouldin Index")
for c in range(maxNClusters):
   print('{:.0f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c], Davies_Bouldin[c]))

# plt.plot(nClusters, TotalWCSS, linewidth = 2, marker = 'o')
# plt.grid(True)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Total WCSS")
# plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
# plt.show()


plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()   

# plt.plot(nClusters, Calinski_Harabasz, linewidth = 2, marker = 'o')
# plt.grid(True)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Calinski-Harabasz Score")
# plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
# plt.show()

plt.plot(nClusters, Davies_Bouldin, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(numpy.arange(1, maxNClusters+1, step = 1))
plt.show()   
