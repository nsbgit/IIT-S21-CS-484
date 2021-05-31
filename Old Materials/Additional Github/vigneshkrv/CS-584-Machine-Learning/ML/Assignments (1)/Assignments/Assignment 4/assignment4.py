import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
import numpy as np

TreasuryRate = pandas.read_csv('ChicagoDiabetes.csv',delimiter=',')


X = TreasuryRate[['Crude Rate 2000', 'Crude Rate 2001', 'Crude Rate 2002', 'Crude Rate 2003',
                  'Crude Rate 2004', 'Crude Rate 2005', 'Crude Rate 2006', 'Crude Rate 2007', 'Crude Rate 2008',
                  'Crude Rate 2009', 'Crude Rate 2010', 'Crude Rate 2011']]

nObs = X.shape[0]
nVar = X.shape[1]

print(X.head)

#pandas.plotting.scatter_matrix(X, figsize=(20,20), c = 'red',
#                               diagonal='hist', hist_kwds={'color':['burlywood']})
#plt.suptitle("Scatter plot matrix for the variables")

# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)

#print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

#print('Explained Variance: \n', _thisPCA.explained_variance_)
#print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
#print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
#print('Principal Components: \n', _thisPCA.components_)

#plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
#plt.title("Explained Variances against their indices")
#plt.xlabel('Index')
#plt.ylabel('Explained Variance Ratio')
#plt.xticks(numpy.arange(0,nVar))
#plt.axhline((1/nVar), color = 'r', linestyle = '--')
#plt.grid(True)
#plt.show()

#cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
#plt.title("Cumulative Sum of Explained Variances against their indices")
#plt.plot(cumsum_variance_ratio, marker = 'o')
#plt.xlabel('Index')
#plt.ylabel('Cumulative Explained Variance Ratio')
#plt.xticks(numpy.arange(0,nVar))
#plt.grid(True)
#plt.show()

first2PC = _thisPCA.components_[:, [0,1]]
#print('Principal COmponent: \n', first2PC)

# Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 15

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

#Draw the Elbow and the Silhouette charts  
   
#plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
#plt.title("Elbow Chart")
#plt.grid(True)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Elbow Value")
#plt.xticks(numpy.arange(2, maxNClusters, 1))
#plt.show()

#plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
#plt.title("Silhouette Chart")
#plt.grid(True)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Silhouette Value")
#plt.xticks(numpy.arange(2, maxNClusters, 1))
#plt.show()
   
# Fit the 4 cluster solution'
kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_

print("\ncount\n", X_transformed['Cluster ID'].value_counts())

TreasuryRate['tp2000']=''
TreasuryRate['tp2000']=(TreasuryRate['Hospitalizations 2000'] / TreasuryRate['Crude Rate 2000'])*10000
TreasuryRate['tp2001']=''
TreasuryRate['tp2001']=(TreasuryRate['Hospitalizations 2001'] / TreasuryRate['Crude Rate 2001'])*10000
TreasuryRate['tp2002']=''
TreasuryRate['tp2002']=(TreasuryRate['Hospitalizations 2002'] / TreasuryRate['Crude Rate 2002'])*10000
TreasuryRate['tp2003']=''
TreasuryRate['tp2003']=(TreasuryRate['Hospitalizations 2003'] / TreasuryRate['Crude Rate 2003'])*10000
TreasuryRate['tp2004']=''
TreasuryRate['tp2004']=(TreasuryRate['Hospitalizations 2004'] / TreasuryRate['Crude Rate 2004'])*10000
TreasuryRate['tp2005']=''
TreasuryRate['tp2005']=(TreasuryRate['Hospitalizations 2005'] / TreasuryRate['Crude Rate 2005'])*10000
TreasuryRate['tp2006']=''
TreasuryRate['tp2006']=(TreasuryRate['Hospitalizations 2006'] / TreasuryRate['Crude Rate 2006'])*10000
TreasuryRate['tp2007']=''
TreasuryRate['tp2007']=(TreasuryRate['Hospitalizations 2007'] / TreasuryRate['Crude Rate 2007'])*10000
TreasuryRate['tp2008']=''
TreasuryRate['tp2008']=(TreasuryRate['Hospitalizations 2008'] / TreasuryRate['Crude Rate 2008'])*10000
TreasuryRate['tp2009']=''
TreasuryRate['tp2009']=(TreasuryRate['Hospitalizations 2009'] / TreasuryRate['Crude Rate 2009'])*10000
TreasuryRate['tp2010']=''
TreasuryRate['tp2010']=(TreasuryRate['Hospitalizations 2010'] / TreasuryRate['Crude Rate 2010'])*10000
TreasuryRate['tp2011']=''
TreasuryRate['tp2011']=(TreasuryRate['Hospitalizations 2011'] / TreasuryRate['Crude Rate 2011'])*10000

TreasuryRate['Cluster ID'] = X_transformed['Cluster ID']

cluster0 = TreasuryRate[TreasuryRate['Cluster ID'] == 0]
cluster1 = TreasuryRate[TreasuryRate['Cluster ID'] == 1]
cluster2 = TreasuryRate[TreasuryRate['Cluster ID'] == 2]
cluster3 = TreasuryRate[TreasuryRate['Cluster ID'] == 3]

toPlot = pandas.DataFrame()
toPlot['Years'] = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
toPlot['cluster0']= ''
years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
i=0
#diab0.reset_index()
i=0
for year in years:
    tp = sum(cluster0['tp'+year])
    hosp = sum(cluster0['Hospitalizations '+year])
    toPlot.loc[i,'cluster0'] =  (hosp/tp)*10000
    i=i+1
        
toPlot['cluster1']= ''
#diab1.reset_index()
i=0
for year in years:
    tp = sum(cluster1['tp'+year])
    hosp = sum(cluster1['Hospitalizations '+year])
    toPlot.loc[i,'cluster1'] =  (hosp/tp)*10000
    i=i+1

toPlot['cluster2']= ''
#diab2.reset_index()
i=0
for year in years:
    tp = sum(cluster2['tp'+year])
    hosp = sum(cluster2['Hospitalizations '+year])
    toPlot.loc[i,'cluster2'] =  (hosp/tp)*10000
    i=i+1

toPlot['cluster3']= ''
#diab2.reset_index()
i=0
for year in years:
    tp = sum(cluster3['tp'+year])
    hosp = sum(cluster3['Hospitalizations '+year])
    toPlot.loc[i,'cluster3'] =  (hosp/tp)*10000
    i=i+1 


toPlot['ref']=['25.4','25.8','27.2','25.4','26.2','26.6','27.4','28.7','27.9','27.5','26.8','25.6']
toPlot['ref']=toPlot['ref'].astype(float)

plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
plt.title("Crude hospitalization rates in each cluster against the years")
plt.plot( 'Years', 'cluster0', data=toPlot, color='black',linewidth=4)
plt.plot( 'Years', 'cluster1', data=toPlot, color='cyan',linewidth=4)
plt.plot( 'Years', 'cluster2', data=toPlot, marker='', color='magenta', linewidth=4)
plt.plot( 'Years', 'cluster3', data=toPlot, marker='', color='yellow', linewidth=4)
plt.plot( 'Years', 'ref', data=toPlot, linestyle=':', color='blue',markersize = 14, linewidth=4, label = 'Reference')
plt.xlabel('Year')
plt.ylabel('Crude Rate (per 10,000)')
plt.legend(loc = 'center right', fontsize = 'xx-large')
plt.grid()
plt.show()