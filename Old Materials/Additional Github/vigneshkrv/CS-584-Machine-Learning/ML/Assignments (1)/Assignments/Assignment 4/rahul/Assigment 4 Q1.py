import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics


diab = pandas.read_csv('ChicagoDiabetes.csv',delimiter=',')

diab.shape
diab.columns
z=pandas.DataFrame()

X = diab[['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003',
          'Crude Rate 2004', 'Crude Rate 2005', 'Crude Rate 2006', 'Crude Rate 2007',
          'Crude Rate 2008','Crude Rate 2009', 'Crude Rate 2010', 'Crude Rate 2011']]

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

plt.plot(_thisPCA.explained_variance_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)

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

diab['tp2000']=''
diab['tp2000']=(diab['Hospitalizations 2000'] / diab['Crude Rate 2000'])*100
diab['tp2001']=''
diab['tp2001']=(diab['Hospitalizations 2001'] / diab['Crude Rate 2001'])*100
diab['tp2002']=''
diab['tp2002']=(diab['Hospitalizations 2002'] / diab['Crude Rate 2002'])*100
diab['tp2003']=''
diab['tp2003']=(diab['Hospitalizations 2003'] / diab['Crude Rate 2003'])*100
diab['tp2004']=''
diab['tp2004']=(diab['Hospitalizations 2004'] / diab['Crude Rate 2004'])*100
diab['tp2005']=''
diab['tp2005']=(diab['Hospitalizations 2005'] / diab['Crude Rate 2005'])*100
diab['tp2006']=''
diab['tp2006']=(diab['Hospitalizations 2006'] / diab['Crude Rate 2006'])*100
diab['tp2007']=''
diab['tp2007']=(diab['Hospitalizations 2007'] / diab['Crude Rate 2007'])*100
diab['tp2008']=''
diab['tp2008']=(diab['Hospitalizations 2008'] / diab['Crude Rate 2008'])*100
diab['tp2009']=''
diab['tp2009']=(diab['Hospitalizations 2009'] / diab['Crude Rate 2009'])*100
diab['tp2010']=''
diab['tp2010']=(diab['Hospitalizations 2010'] / diab['Crude Rate 2010'])*100
diab['tp2011']=''
diab['tp2011']=(diab['Hospitalizations 2011'] / diab['Crude Rate 2011'])*100

diab.columns

diab['Cluster ID'] = X_transformed['Cluster ID']

diab0 = diab[diab['Cluster ID'] == 0]
diab0.shape

diab1 = diab[diab['Cluster ID'] == 1]

diab2 = diab[diab['Cluster ID'] == 2]

diab3 = diab[diab['Cluster ID'] == 3]

fcr = pandas.DataFrame()
fcr['Years'] = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
fcr['0']= ''
years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
i=0
diab0.reset_index()
i=0
for year in years:
    tp = sum(diab0['tp'+year])
    hosp = sum(diab0['Hospitalizations '+year])
    fcr.loc[i,'0'] =  (hosp/tp)*100
    i=i+1
        
fcr['1']= ''
years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
diab1.reset_index()
i=0
for year in years:
    tp = sum(diab1['tp'+year])
    hosp = sum(diab1['Hospitalizations '+year])
    fcr.loc[i,'1'] =  (hosp/tp)*100
    i=i+1

fcr['2']= ''
years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
diab2.reset_index()
i=0
for year in years:
    tp = sum(diab2['tp'+year])
    hosp = sum(diab2['Hospitalizations '+year])
    fcr.loc[i,'2'] =  (hosp/tp)*100
    i=i+1

fcr['3']= ''
years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']
diab2.reset_index()
i=0
for year in years:
    tp = sum(diab3['tp'+year])
    hosp = sum(diab3['Hospitalizations '+year])
    fcr.loc[i,'3'] =  (hosp/tp)*100
    i=i+1 


fcr['4']=['25.4','25.8','27.2','25.4','26.2','26.6','27.4','28.7','27.9','27.5','26.8','25.6']
fcr['4']=fcr['4'].astype(float)

plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
plt.plot( 'Years', '0', data=fcr, color='orange',linewidth=4)
plt.plot( 'Years', '1', data=fcr, color='yellow',linewidth=4)
plt.plot( 'Years', '2', data=fcr, marker='', color='green', linewidth=4)
plt.plot( 'Years', '3', data=fcr, marker='', color='red', linewidth=4)
plt.plot( 'Years', '4', data=fcr, marker='o', color='blue',markersize = 14, linewidth=4, label = 'Avg')
plt.xlabel('Year')
plt.ylabel('Crude Rate (per 10,000)')
plt.legend(loc = 'center right', fontsize = 'xx-large')
plt.grid()
plt.show()

#Vaibhav's Plot
plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
plt.plot( 'Years', '0', data=fcr, color='orange',linewidth=4)
plt.plot( 'Years', '1', data=fcr, color='yellow',linewidth=4)
plt.plot( 'Years', '2', data=fcr, marker='', color='green', linewidth=4)
plt.plot( 'Years', '3', data=fcr, marker='', color='red', linewidth=4)
plt.plot( 'Years', '4', data=fcr,linestyle = 'dotted', color='skyblue',markersize = 14, linewidth=8, label = 'Avg')
plt.xlabel('Year')
plt.ylabel('Crude Rate (per 10,000)')
plt.legend(loc = 'top right', fontsize = 'x-large')
plt.grid()
plt.show()

#Russi's Plot
plt.plot( 'Years', '0', data=fcr, color='purple',linewidth=4)
plt.plot( 'Years', '1', data=fcr, color='green',linewidth=4)
plt.plot( 'Years', '2', data=fcr, marker='', color='violet', linewidth=4)
plt.plot( 'Years', '3', data=fcr, marker='', color='red', linewidth=4)
plt.plot( 'Years', '4', data=fcr,linestyle = 'dotted', color='skyblue',markersize = 14, linewidth=8, label = 'Avg')
plt.xlabel('Year')
plt.ylabel('Crude Rate (per 10,000)')
plt.legend(loc = 'top right', fontsize = 'large')
plt.grid()
plt.show()







