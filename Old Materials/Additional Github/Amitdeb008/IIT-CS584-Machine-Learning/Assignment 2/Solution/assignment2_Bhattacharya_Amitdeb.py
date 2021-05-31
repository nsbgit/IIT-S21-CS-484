#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[108]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.neighbors
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
get_ipython().run_line_magic('matplotlib', 'inline')


# Question 1

# In[109]:


#Loading the dataset
df = pd.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 2\\Groceries.csv')


# In[110]:


#Top few rows of the dataset
df.head()


# a)	(10 points) Create a dataset which contains the number of distinct items in each customer’s market basket. Draw a histogram of the number of unique items.  What are the median, the 25th percentile and the 75th percentile in this histogram?

# In[111]:


dataset = df.groupby(['Customer'])['Item'].count()
dataset = dataset.sort_values()


# In[112]:


median, q1, q3 = np.percentile(dataset, 50), np.percentile(dataset, 25),np.percentile(dataset, 75)


# In[113]:


plt.hist(dataset)
plt.axvline(q1, color='red', alpha=.9, linewidth=.9)
plt.axvline(q3, color='red', alpha=.9, linewidth=.9)
plt.axvline(median, color='orange', alpha=.9, linewidth=.9)
plt.grid(True)
plt.title("Histogram of Unique Items")
plt.ylabel("Frequency")
plt.show()


# In[114]:


print("Median: {}, 25th Percentile: {}, 75th Percentile: {}".format(median,q1,q3))


# b)	(10 points) If you are interested in the k-itemsets which can be found in the market baskets of at least seventy five (75) customers.  How many itemsets can you find?  Also, what is the largest k value among your itemsets?

# In[115]:


itemcust = df.groupby(['Customer'])['Item'].count()
ListItem = df.groupby(['Customer'])['Item'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary,columns=te.columns_)
min_sup = 75/len(itemcust)
frequent_itemsets = apriori(ItemIndicator,min_support=min_sup,use_colnames=True,max_len=32)
kitemsetmax = len(frequent_itemsets['itemsets'][len(frequent_itemsets)-1])


# In[116]:


frequent_itemsets['itemsets']


# In[117]:


print("Total Item-sets Found: {}\nThe highest k-value is : {}".format(frequent_itemsets.shape[0],kitemsetmax))


# c)	(10 points) Find out the association rules whose Confidence metrics are at least 1%.  How many association rules have you found?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.  Also, you do not need to show those rules.

# In[118]:


# Convert the data to the Item List format
ListItem = df.groupby(['Customer'])['Item'].apply(list).values.tolist()
# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)


# In[119]:


# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)


# In[120]:


print("Total Association rules found: {}" .format(assoc_rules.shape[0]))


# d)	(10 points) Graph the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules you found in (c).  Please use the Lift metrics to indicate the size of the marker. 

# In[121]:


plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()


# e)	(10 points) List the rules whose Confidence metrics are at least 60%.  Please include their Support and Lift metrics.

# In[122]:


# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)


# In[123]:


assoc_rules


# Question 2

# In[124]:


#Importing the dataset
spiral = pd.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 2\\Spiral.csv')


# In[125]:


#Top few rows of the dataframe
spiral.head()


# a)	(10 points) Generate a scatterplot of y (vertical axis) versus x (horizontal axis).  How many clusters will you say by visual inspection?

# In[126]:


#Scatterplot of  and Y values
plt.scatter(x=spiral['x'],y=spiral['y'])
plt.show()


# b)	(10 points) Apply the K-mean algorithm directly using your number of clusters that you think in (a). Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?

# In[127]:


trainData = spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=60616).fit(trainData)
spiral['KMeanCluster'] = kmeans.labels_
for i in range (2):
    spiral.loc[spiral['KMeanCluster'] == i]
    
plt.scatter(spiral['x'], spiral['y'], c = spiral['KMeanCluster'] )
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# c)	(10 points) Apply the nearest neighbor algorithm using the Euclidean distance.  How many nearest neighbors will you use?  Remember that you may need to try a couple of values first and use the eigenvalue plot to validate your choice.

# In[128]:


kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)

# Create the Adjacency and the Degree matrices
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
        
Lmatrix = Degree - Adjacency

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()


# d)	(10 points) Retrieve the first two eigenvectors that correspond to the first two smallest eigenvalues.  Display up to ten decimal places the means and the standard deviation of these two eigenvectors.  Also, plot the first eigenvector on the horizontal axis and the second eigenvector on the vertical axis.

# In[129]:


# Inspect the values of the selected eigenvectors 
Z = evecs[:,[0,1]]

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()


# e)	(10 points) Apply the K-mean algorithm on your first two eigenvectors that correspond to the first two smallest eigenvalues. Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?

# In[130]:


kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(Z)

spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(spiral['x'], spiral['y'], c = spiral['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

