import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics

inputData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\ChicagoCompletedPotHole.csv',
                            delimiter=',',
                            usecols=['N_POTHOLES_FILLED_ON_BLOCK', 'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE'])

# Transform two variables
inputData['LOG_N_POTHOLES_FILLED_ON_BLOCK'] = numpy.log(inputData['N_POTHOLES_FILLED_ON_BLOCK'])
inputData['LOG1P_N_DAYS_FOR_COMPLETION'] = numpy.log1p(inputData['N_DAYS_FOR_COMPLETION'])

# Print number of missing values per variable
print('Number of Missing Values:')
print(pandas.Series.sort_index(inputData.isna().sum()))

# Create the training data
trainData = (inputData[['LOG_N_POTHOLES_FILLED_ON_BLOCK', 'LOG1P_N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE']]).dropna()

nObs = trainData.shape[0]

maxNClusters = 10

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
Calinski = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   _thisKmeans = cluster.KMeans(n_clusters = KClusters, random_state = 20201014)
   _thisCluster = _thisKmeans.fit(trainData)

   # Specify the sample_size to the largest integer that will not trigger the memory error for the 2-cluster
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(trainData, _thisCluster.labels_)
       Calinski[c] = metrics.calinski_harabasz_score(trainData, _thisCluster.labels_)
   else:
       Silhouette[c] = numpy.NaN
       Calinski[c] = numpy.NaN

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = _thisCluster.labels_[i]
      nC[k] += 1
      diff = trainData.iloc[i,] - _thisCluster.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Elbow Value\t Silhouette Value\t Calinski-Harabasz Score:")
for c in range(maxNClusters-1):
   print('{:.0f} \t {:.7f} \t {:.7f} \t {:.7f}'
         .format(nClusters[c], Elbow[c], Silhouette[c], Calinski[c]))

# Draw the Elbow chart  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters+1, 1))
plt.yticks(numpy.arange(2.7,3.3,0.1))
plt.show()

# Draw the Silhouette chart
plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters+1, 1))
plt.yticks(numpy.arange(0.37,0.43,0.01))
plt.show()

# Draw the Calinski-Harabasz Score chart
plt.plot(nClusters, Calinski, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.xticks(numpy.arange(2, maxNClusters+1, 1))
plt.show()
