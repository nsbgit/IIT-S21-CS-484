import numpy
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import pandas
import matplotlib.pyplot as plt
import math

trainData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\cars.csv',
                       delimiter=',', usecols=['Horsepower', 'Weight'])

nCars = trainData.shape[0]

# Part (a)
nClusters = numpy.zeros(15)
Elbow = numpy.zeros(15)
Silhouette = numpy.zeros(15)
TotalWCSS = numpy.zeros(15)
Inertia = numpy.zeros(15)

for c in range(15):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=0).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nCars):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   print("Cluster Assignment:", kmeans.labels_)
   for k in range(KClusters):
      print("Cluster ", k)
      print("Centroid = ", kmeans.cluster_centers_[k])
      print("Size = ", nC[k])
      print("Within Sum of Squares = ", WCSS[k])
      print(" ")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(15):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Part (b)
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.show()
