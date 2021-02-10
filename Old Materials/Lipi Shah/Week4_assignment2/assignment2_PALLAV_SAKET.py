## Note:- i have taken referance from sample code of lecture slides for some part of code.
########################### Import Statements #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import DistanceMetric as DM
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

########################### Question 1 (a)#####################################

groceriesData = pd.read_csv('D:\\IIT Edu\\Sem1\\MachineLearning\\Week4_assignment2\\Groceries.csv', delimiter=',')

data_item = groceriesData.groupby(['Customer'])['Item'].nunique() #distinct dataset for each customer 
#print(data_item)
cust_freq= groceriesData.Customer.value_counts()
sorted_cust_Freq=sorted(cust_freq)
itemset = Counter(sorted_cust_Freq)

updatedData=pd.DataFrame.from_dict(itemset, orient='index').reset_index()
updatedData=updatedData.rename(columns={'index':'Itemset', 0:'Customers'})

print("unique itemset frequency table: \n",updatedData)


###Histogram
plt.hist(cust_freq)
plt.title("Histogram of number of unique items")
plt.xlabel("Unique items")
plt.ylabel("Customer")
plt.grid(axis="x")
plt.show()

#updatedData.describe()
#25 percentile, median and 75 percentile
median =np.median(cust_freq)
print("Median",median)
LowerQuartile=np.percentile(cust_freq,25)
print("LowerQuartile",LowerQuartile)
UpperQuartile=np.percentile(cust_freq,75)
print("UpperQuartile",UpperQuartile)

########################### Question 1 (b)#####################################
# Sale Receipt data to the Item List
ListItem = groceriesData.groupby(['Customer'])['Item'].apply(list).values.tolist()
print(ListItem)

#Item List to Item Indicator
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
print(ItemIndicator)

totalTransactions=np.count_nonzero(data_item)
minSupport=75/totalTransactions
freq_itemsets = apriori(ItemIndicator, min_support = minSupport, use_colnames = True) # Frequent itemsets
print("Frequent itemset \n",freq_itemsets.head())
noOfItemset=freq_itemsets.support.count()
print("Total number of itemset: ",noOfItemset,"\n")
#freq_itemsets.head()
k = len(freq_itemsets['itemsets'] [len(freq_itemsets)-1])
print("Largest value of K = ",k)


########################### Question 1 (c)#####################################

# association rules with Confidence metrics at least 1%.  
associationRules = association_rules(freq_itemsets, metric = "confidence", min_threshold = 0.01)
print("Association rules \n",associationRules)
print(associationRules)
#associationRules.describe()

noOfassociationrules=associationRules.antecedents.count()
print("Total number of association rules: ",noOfassociationrules)

########################### Question 1 (d)#####################################

plt.figure(figsize=(8, 6))
plt.scatter(associationRules['confidence'], associationRules['support'], s = associationRules['lift'])
plt.grid(True)
plt.title("Confidence metrics vs. Support metrics")
plt.xlabel("Confidence")
plt.ylabel("Support")
cbar=plt.colorbar()
cbar.set_label("Lift")
plt.show()


########################### Question 1 (e)#####################################

associationRules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.6)
#associationRules
associationRules[['conviction','antecedents','consequents','support','lift']]


########################### Question 2 (a)#####################################

spiralData = pd.read_csv('D:\\IIT Edu\\Sem1\\MachineLearning\\Week4_assignment2\\Spiral.csv',
                         delimiter=',')

plt.figure(figsize=(8, 6))
plt.scatter(spiralData['x'], spiralData['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

print("By visual inspection we can say that there are 2 clusters")

########################### Question 2 (b)#####################################

trainData = spiralData[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=60616).fit(trainData)

#print("Cluster Centroids = \n", kmeans.cluster_centers_)

spiralData['KMeanCluster'] = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(spiralData['x'], spiralData['y'], c = spiralData['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

########################### Question 2 (c)#####################################

kNNSpec = kNN(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

########################### Question 2 (d)#####################################

# getting the distances among the observations
distObject = DM.get_metric('euclidean')
distances = distObject.pairwise(trainData)

nObs = spiralData.shape[0]
#Adjacency and the Degree matrices
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

print("Adjacency Matrix: ",Adjacency)
print("Degree Matrix: ",Degree)

Lmatrix = Degree - Adjacency

print("Laplace Matrix: \n",Lmatrix)

evals, evecs = LA.eigh(Lmatrix)

print("Eigenvalues of Laplace Matrix = \n", evals,"\n")

print("Eigenvectors of Laplace Matrix = \n",evecs,"\n")

plt.figure(figsize=(8, 6))
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

Z = evecs[:,[0,1]]
#Z
print('    Mean',"                     Std")
print(Z[[0]].mean(), Z[[0]].std())
print(Z[[1]].mean(), Z[[1]].std())

plt.figure(figsize=(8, 6))
plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.grid()
plt.show()

########################### Question 2 (e)#####################################

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(Z)
spiralData['SpectralCluster'] = kmeans_spectral.labels_
plt.figure(figsize=(8, 6))
plt.scatter(spiralData['x'], spiralData['y'],c = spiralData['SpectralCluster'])
plt.xlabel('X Observation')
plt.ylabel('Y Observation')
plt.grid(True)
plt.show()