import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model

# The following data for all US Treasury Rate since 1/1/2018

TreasuryRate = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\TreasuryRate.csv',
                          delimiter=',')

# Feature variables
X = TreasuryRate[['Bill_1Month', 'Bill_3Month', 'Bill_6Month', 'Bill_1Year',
                  'Note_2Year', 'Note_3Year', 'Note_5Year', 'Note_7Year', 'Note_10Year',
                  'Bond_20Year', 'Bond_30Year']]

nObs = X.shape[0]
nVar = X.shape[1]

pandas.plotting.scatter_matrix(X, figsize=(20,20), c = 'red',
                               diagonal='hist', hist_kwds={'color':['burlywood']})

# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)

print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

print('Explained Variance: \n', _thisPCA.explained_variance_)
print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
print('Principal Components: \n', _thisPCA.components_)

plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)

# Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20181010).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters-1):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

# Fit the 4 cluster solution'
kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_

# Draw the first two PC using cluster label as the marker color 
carray = ['red', 'orange', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = X_transformed[X_transformed['Cluster ID'] == i]
    plt.scatter(x = subData[0],
                y = subData[1], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.axis(aspect = 'equal')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis(aspect = 'equal')
plt.legend(title = 'Cluster ID', fontsize = 12, markerscale = 2)
plt.show()

