# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:49:43 2020

@author: Lipi
"""
## Note:- I have taken reference from the professor sample code for some part of assignment.
########################### Import Statements #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sklearn.cluster as cluster
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import DistanceMetric as DM
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.stats import variation  
import numpy as np 
import scipy

#################### ANSWER 1 A#######################################3333333
print("#################### ANSWER 1 A #######################################")
groceriesData = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4_assignment2\\Groceries.csv', delimiter=',')

Unique_Items_per_customer = groceriesData.groupby('Customer')['Item'].apply(lambda x: x.unique().shape[0])

Hist_Input = groceriesData.groupby('Item').Customer.count()

Hist_Input.columns = ['items','count']
Hist_Input.index = range(1,len(Hist_Input)+1)
plt.hist(Unique_Items_per_customer)
plt.title("Histogram of number of unique items")
plt.xlabel("Unique items")
plt.ylabel("Customer")
plt.grid(axis="x")
plt.show()
Unique_Items_per_customer.describe()
print(Unique_Items_per_customer.describe())
#question 1 B
print("#################### ANSWER 1 B #######################################")
# Convert the Sale Receipt data to the Item List format
ListItem = groceriesData.groupby(['Customer'])['Item'].apply(list).values.tolist()
# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(ItemIndicator, min_support = 75/Unique_Items_per_customer.count(), use_colnames = True)
print("No of K-item sets",len(frequent_itemsets))
print("Largest value of K =",len(frequent_itemsets['itemsets'].max()))
# Question 1 C
print("#################### ANSWER 1 C #######################################")
# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
assoc_rules.count()
ass_c_one = min([assoc_rules.count()['consequents'],assoc_rules.count()['antecedents']])
print("Total no of Asscociation rule at confidence >= 1% =" ,min([assoc_rules.count()['consequents'],assoc_rules.count()['antecedents']]))

print("#################### ANSWER 1 D #######################################")
#question 1 D
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'], marker = 's')
plt.colorbar()
plt.grid(True)
plt.title("Confidence Vs. Support")
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

print("#################### ANSWER 1 E #######################################")
#question 1 E
from tabulate import tabulate
required_data = assoc_rules[assoc_rules['confidence']>=0.6]

for index, row in required_data.iterrows():
    print(tabulate([[row['antecedents'],row['consequents'], row['support'], row['lift']]], headers=['antecedents', 'consequents', 'support', 'lift']))
    
    
#Question 2 
print("##################### Question 2 #################################")
print("##################### Question 2 A #################################")
cars = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4_assignment2\\cars.csv',delimiter=',')
a = cars.Type.values
u,indices = np.unique(a,return_counts = True)
df_Cat_Type = pd.DataFrame(u)
df_Cat_Type[1] = pd.DataFrame(indices)
df_Cat_Type.columns = ['Type', 'Frequency']
#print("Total Count of Category = 'Type' is ",cars.Type.count())
print("Frequencies of each Type Categories")
print(df_Cat_Type)

print("##################### Question 2 B #################################")
#DriveTrain
b = cars.DriveTrain.values
u_DriveTrain,indices_DriveTrain = np.unique(b,return_counts = True)
df_Cat_DriveTrain = pd.DataFrame(u_DriveTrain)
df_Cat_DriveTrain[1] = pd.DataFrame(indices_DriveTrain)
df_Cat_DriveTrain.columns = ['DriveTrain', 'Frequency']
#print("Total Count of Category = 'DriveTrain' is ",cars.DriveTrain.count())
print("Frequencies of each DriveTrain Categories")
print(df_Cat_DriveTrain)

print("##################### Question 2 C #################################")
#Origin
c = cars.Origin.values
u_Origin,indices_Origin = np.unique(c,return_counts = True)
df_Cat_Origin = pd.DataFrame(u_Origin)
df_Cat_Origin[1] = pd.DataFrame(indices_Origin)
df_Cat_Origin.columns = ['Origin', 'Frequency']
car_origin_asia = cars[ cars["Origin"]=="Asia"].count(axis=1)
len_origin_asia = car_origin_asia.count()

car_origin_Europe = cars[ cars["Origin"]=="Europe"].count(axis=1)
len_origin_Europe = car_origin_Europe.count()

print("The distance between Origin = ‘Asia’ and Origin = ‘Europe’ =",
      abs((1/len_origin_asia) + (1/len_origin_Europe) ))

print("##################### Question 2 D #################################")
car_cylinders_five = cars[ cars["Cylinders"]==5].count(axis=1)
len_car_cylinders_five = car_cylinders_five.count()
#Count for measing value
count_nan = len(cars["Cylinders"]) - cars["Cylinders"].count()
# either one... =>count_nan =cars["Cylinders"].isna().sum()
#print distance
print("The distance between Cylinders = 5 and Cylinders = Missing =",
      abs((1/len_car_cylinders_five) + (1/count_nan) ))

print("################## Question 2 E ###################################")
      
import numpy as np
from scipy.stats import mode 
import numpy as np   


cars.fillna(0, inplace=True)
cars = cars[['Type','Origin','DriveTrain','Cylinders']]

#initialize first 3 cluster centroid with any random data points
c0 = cars.iloc[[0]].to_numpy().ravel()
c1 = cars.iloc[[70]].to_numpy().ravel()
c2 = cars.iloc[[10]].to_numpy().ravel()

c0_prev = []
c1_prev = []
c2_prev = []


for loop in range(1,6):
    m0 = []
    m1 = []
    m2 = []
    for x in range(0, cars.shape[0]):
        c0_prev = c0
        c1_prev = c1
        c2_prev = c2
        min = 20 # Any random value greater than one 
        track = 0
        row = cars.iloc[[x]].to_numpy().ravel()
        #print(row)   ex:row[0] ='Wagon'
        #first cluster loop --> Track each cluster by trackid 0,1,2
        #first cluster --> Track =0
        sum = 0
        for y in range(0,4):
            if c0[y] != row[y]:
                sum+=(1/cars[ cars[cars.columns[y]]==c0[y]].count(axis=1).count())+(1/cars[ cars[cars.columns[y]]==row[y]].count(axis=1).count())
            #print("value of x",x,":",sum)
        if min > sum:
            min = sum
            track = 0
        #second cluster --> Track =1
        sum = 0
        for y in range(0,4):
            if c1[y] != row[y]:
                sum+=(1/cars[ cars[cars.columns[y]]==c1[y]].count(axis=1).count())+(1/cars[ cars[cars.columns[y]]==row[y]].count(axis=1).count())
            #print("value of x",x,":",sum)
        if min > sum:
            min = sum
            track = 1
        #Third cluster --> Track =2
        sum = 0
        for y in range(0,4):
            if c2[y] != row[y]:
                sum+=(1/cars[ cars[cars.columns[y]]==c2[y]].count(axis=1).count())+(1/cars[ cars[cars.columns[y]]==row[y]].count(axis=1).count())
            #print("value of x",x,":",sum)
        if min > sum:
            min = sum
            track = 2
        if track == 0:
            m0.append(row)
        if track == 1:
            m1.append(row)
        if track == 2:
            m2.append(row)
    #getting new centroid c0,c1,c2
    val, count = mode(m0, axis = 0)
    c0 = val.ravel().tolist()
    val, count = mode(m1, axis = 0)
    c1 = val.ravel().tolist()
    val, count = mode(m2, axis = 0)
    c2 = val.ravel().tolist()
       
                   
print("Centroid 0 :", c0, " and observations in this cluster are ",len(m0))
print("centroid 1 :",c1, " and observations in this cluster are ",len(m1))
print("centroid 2 :",c2, " and observations in this cluster are ",len(m2))


print("################## Question 2 F ###################################")
df_m0 = pd.DataFrame(m0)
print("Cluster 0: Europe:", df_m0[ df_m0[1]=="Europe"].count(axis=1).count())
print("Cluster 0: USA:", df_m0[ df_m0[1]=="USA"].count(axis=1).count())
print("Cluster 0: Asia:", df_m0[ df_m0[1]=="Asia"].count(axis=1).count())

df_m1 = pd.DataFrame(m1)
print("Cluster 1: Europe:", df_m1[ df_m1[1]=="Europe"].count(axis=1).count())
print("Cluster 1: USA:", df_m1[ df_m1[1]=="USA"].count(axis=1).count())
print("Cluster 1: Asia:", df_m1[ df_m1[1]=="Asia"].count(axis=1).count())

df_m2 = pd.DataFrame(m2)
print("Cluster 2: Europe:", df_m2[ df_m2[1]=="Europe"].count(axis=1).count())
print("Cluster 2: USA:", df_m2[ df_m2[1]=="USA"].count(axis=1).count())
print("Cluster 2: Asia:", df_m2[ df_m2[1]=="Asia"].count(axis=1).count())
    
                  
print("##################################Question 3##########################")
print("####################### Question 3 A ############################")
FourCircle = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4_assignment2\\FourCircle.csv',delimiter=',')
plt.figure(figsize=(8, 6))
plt.scatter(FourCircle['x'], FourCircle['y'],c = FourCircle['ring'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
print("As per visual inspection there are total 4 Clusters")

print("####################### Question 3 B ############################")

trainData = FourCircle[['x','y']]
kmeans = cluster.KMeans(n_clusters=4, random_state=60616 ).fit(trainData)

FourCircle['KMeanCluster'] = kmeans.labels_
    
plt.figure(figsize=(8, 6))
plt.scatter(FourCircle['x'], FourCircle['y'], c = FourCircle['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

print("####################### Question 3 C ############################")

import math
import sklearn.neighbors

#for x in range(3,12): experimenting with different neighrest neighbors
print("After Cheked with different combination of No of Cluster - No of Neighrest neighbor:",
    "Value of Neighrest neighbour :","6")
# six nearest neighbors
kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 6, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

print("####################### Question 3 d ############################")    
# Retrieve the distances among the observations
distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(trainData)
nObs = FourCircle.shape[0]
# Create the Adjacency and the Degree matrices
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]
print("Adjacency Matrix")
print(Adjacency)
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

print("Degree Matrix")
print(Degree)        
Lmatrix = Degree - Adjacency
print("Lmatrix Matrix")
print(Lmatrix)  
Adjacency_mat = pd.DataFrame(Adjacency)
Degree_mat =  pd.DataFrame(Degree)
Lmatrix_mat =  pd.DataFrame(Lmatrix)
evals, evecs = LA.eigh(Lmatrix)


# Series plot of the smallest twenty eigenvalues to determine the number of neighbors
sequence = np.arange(1,15,1) 
plt.plot(sequence, evals[0:14,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.plot(np.arange(1,10,1), evals[0:9,],marker="o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

print("Six eigenvalues Value",evals[0:6,],"\n")

print("Practically Zero eigenvalues as below:")
for i in range(6):
    if evals[0:6,][i,]<=0.0:
        if evals[0:6,][i,]>-0.02:
            print(evals[i,])

Z = evecs[:,[0,1,2,3]]
print("\n Standard Deviation, Mean and Variance")
print("STD",Z[[0]].mean(), "Mean:", Z[[0]].std(),"Variation:",scipy.stats.variation(Z[0]))
print("STD",Z[[1]].mean(), "Mean:", Z[[1]].std(),"Variation:",scipy.stats.variation(Z[1]))
#print("STD",Z[[2]].mean(), "Mean:", Z[[2]].std(),"Variation:",scipy.stats.variation(Z[2]))
#print("STD",Z[[3]].mean(), "Mean:", Z[[3]].std(),"Variation:",scipy.stats.variation(Z[3]))

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.grid("both")
plt.show()

print("####################### Question 3 e  ############################") 

kmeans_spectral = cluster.KMeans(n_clusters=4, random_state= 60616).fit(Z)

FourCircle['SpectralCluster'] = kmeans_spectral.labels_

plt.figure(figsize=(8, 6))
plt.scatter(FourCircle['x'], FourCircle['y'], c = FourCircle['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

#print("Cluster Centroids = \n", kmeans_spectral.cluster_centers_)
    
