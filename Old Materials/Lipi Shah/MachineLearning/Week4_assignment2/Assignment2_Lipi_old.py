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

#################### ANSWER 1 A#######################################3333333

groceriesData = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4_assignment2\\Groceries.csv', delimiter=',')
#Create a data frame that contains the number of unique items in each customer’s market basket.
#group_customer = groceriesData.groupby(['Customer'])['Item']
#Unique_Items_per_customer = group_customer.nunique()

#Unique_Items_per_customer1 = groceriesData.groupby('Customer').Item.nunique()
Unique_Items_per_customer = groceriesData.groupby('Customer')['Item'].apply(lambda x: x.unique().shape[0])

Hist_Input = groceriesData.groupby('Item').Customer.count()

Hist_Input.columns = ['items','count']
#Hist_Input.columns
#Change Itemnames (Items) to index value
Hist_Input.index = range(1,len(Hist_Input)+1)


#import seaborn as sns, numpy as np
######
#import pandas as pd
#Unique_Items_per_customer = pd.Series(Unique_Items_per_customer)
#ax = sns.distplot(Unique_Items_per_customer)
#
#
#n, bins, patches = plt.hist(x=Unique_Items_per_customer, bins='auto', color='#0504aa',
#                            alpha=0.7, rwidth=0.85)
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('My Very Own Histogram')


plt.hist(Unique_Items_per_customer)
plt.title("Histogram of number of unique items")
plt.xlabel("Unique items")
plt.ylabel("Customer")
plt.grid(axis="x")
plt.show()

Unique_Items_per_customer.describe()

print(Unique_Items_per_customer.describe())
#question 1 B
# Convert the Sale Receipt data to the Item List format
ListItem = groceriesData.groupby(['Customer'])['Item'].apply(list).values.tolist()
# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(ItemIndicator, min_support = 75/Unique_Items_per_customer.count(), use_colnames = True)
#frequent_itemsets.count()['itemsets']
print("No of K-item sets",len(frequent_itemsets))
#frequent_itemsets = sorted(frequent_itemsets['itemsets'])
#frequent_itemsets['itemsets'].max()
print("Largest value of K =",len(frequent_itemsets['itemsets'].max()))

# Question 1 C

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
assoc_rules.count()
print("Total no of Asscociation rule at confidence >= 1% =" ,min([assoc_rules.count()['consequents'],assoc_rules.count()['antecedents']]))


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


#question 1 E
from tabulate import tabulate
required_data = assoc_rules[assoc_rules['confidence']>=0.6]

for index, row in required_data.iterrows():
    print(tabulate([[row['antecedents'],row['consequents'], row['support'], row['lift']]], headers=['antecedents', 'consequents', 'support', 'lift']))
    #print(row['antecedents'],row['consequents'], row['support'], row['lift'])


#from prettytable import PrettyTable
#t = PrettyTable(['antecedents', 'consequents', 'support', 'lift'])
#for index, row in required_data.iterrows():
#    t.add_row([row['antecedents'],row['consequents'], row['support'], row['lift']])
#    print(t)
    
    
#Question 2 A
#What are the frequencies of the categorical feature Type?
#https://www.tutorialspoint.com/numpy/numpy_unique.htm
print("##################### Question 2 #################################")
print("##################### Question 2 A #################################")
cars = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4_assignment2\\cars.csv',delimiter=',')
a = cars.Type.values
u,indices = np.unique(a,return_counts = True)
df_Cat_Type = pd.DataFrame(u)
df_Cat_Type[1] = pd.DataFrame(indices)
df_Cat_Type.columns = ['Type', 'Frequency']
print("Total Count of Category = 'Type' is ",cars.Type.count())
print("Frequencies of each Type Categories")
print(df_Cat_Type)
#df.add_suffix('_X')
#df.rename(columns={'0': 'a', '1': 'c'})
#
#values = np.zeros(20, dtype='int64')
#index = ['Row'+str(i) for i in range(1, len(values)+1)]
#df = pd.DataFrame(values)

print("##################### Question 2 B #################################")
#DriveTrain
#https://note.nkmk.me/en/python-pandas-dataframe-rename/
b = cars.DriveTrain.values
u_DriveTrain,indices_DriveTrain = np.unique(b,return_counts = True)
df_Cat_DriveTrain = pd.DataFrame(u_DriveTrain)
df_Cat_DriveTrain[1] = pd.DataFrame(indices_DriveTrain)
df_Cat_DriveTrain.columns = ['DriveTrain', 'Frequency']
print("Total Count of Category = 'DriveTrain' is ",cars.DriveTrain.count())
print("Frequencies of each DriveTrain Categories")
print(df_Cat_DriveTrain)

print("##################### Question 2 C #################################")
#Origin
c = cars.Origin.values
u_Origin,indices_Origin = np.unique(c,return_counts = True)
df_Cat_Origin = pd.DataFrame(u_Origin)
df_Cat_Origin[1] = pd.DataFrame(indices_Origin)
df_Cat_Origin.columns = ['Origin', 'Frequency']
print("Total Count of Category = 'Origin' is ",cars.Origin.count())
print("Frequencies of each Origin Categories")
print(df_Cat_Origin)

#len(cars.where(cars["Origin"]=="Asia")  )
#len(filter(cars["Origin"]=="Asia", cars))
#one=sum(1 for item in cars if cars["Origin"]=="Asia")
car_origin_asia = cars[ cars["Origin"]=="Asia"].count(axis=1)
len_origin_asia = car_origin_asia.count()

car_origin_Europe = cars[ cars["Origin"]=="Europe"].count(axis=1)
len_origin_Europe = car_origin_Europe.count()

print("The distance between Origin = ‘Asia’ and Origin = ‘Europe’ =",
      abs((1/len_origin_asia) + (1/len_origin_Europe) ))
#
#asian_fre = df_Cat_Origin[ df_Cat_Origin["Origin"]=="Asia"]["Frequency"]
#print(1/asian_fre)
#
#europe_fre = df_Cat_Origin[ df_Cat_Origin["Origin"]=="Europe"]["Frequency"]
#print(1/europe_fre)
#
#print( (1/asian_fre)- (1/europe_fre))
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
d = cars.Cylinders.values
u_Cylinders,indices_Cylinders = np.unique(d,return_counts = True)
df_Cat_Cylinders = pd.DataFrame(u_Cylinders)
df_Cat_Cylinders[1] = pd.DataFrame(indices_Cylinders)
df_Cat_Cylinders.columns = ['Cylinders', 'Frequency']
print("Total Count of Category = 'Cylinders' is ",cars.Cylinders.count())
print("Frequencies of each Cylinders Categories")
print(df_Cat_Cylinders)

conv_cars = cars.apply(lambda x: x.map(1/x.value_counts()))
conv_cars_final = conv_cars[['Type','Origin','DriveTrain','Cylinders']]
#apply k means on conv_cars_final matrix now

print("################## Question 2 E ###################################")

import numpy as np
from kmodes.kmodes import KModes

from scipy.stats import mode 
import numpy as np 

#cars1= cars
#cars1.fillna(cars.mean(), inplace=True)
#cars1 = cars[['Type','Origin','DriveTrain','Cylinders']]
#km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
#clusters = km.fit_predict(cars1)
#print(km.cluster_centroids_)

cars = cars[['Type','Origin','DriveTrain','Cylinders']]

#initialize first 3 cluster centroid
c0 = cars.iloc[[0]].to_numpy().ravel()
c1 = cars.iloc[[10]].to_numpy().ravel()
c2 = cars.iloc[[75]].to_numpy().ravel()




for loop3 in range(1,3):
    m0 = []
    m1 = []
    m2 = []
    for x in range(0, cars.shape[0]):
        
        min = 20 # Any random value greater than zero 
        track = 0
        row = cars.iloc[[x]].to_numpy().ravel()
        #print(row)   ex:row[0] ='Wagon'
        #first cluster loop --> Track each cluster by trackid 0,1,2
        #first cluster --> Track =0
        sum = 0
        for y in range(0,4):
            #sum = 0
            if c0[y] != row[y]:
                sum+=(1/cars[ cars[cars.columns[y]]==c0[y]].count(axis=1).count())+(1/cars[ cars[cars.columns[y]]==row[y]].count(axis=1).count())
            print("value of x first",x,":",sum)
            if min > sum:
                min = sum
                track = 0
        #second cluster --> Track =1
        sum = 0
        for y in range(0,4):
            #sum = 0
            if c1[y] != row[y]:
                sum+=(1/cars[ cars[cars.columns[y]]==c1[y]].count(axis=1).count())+(1/cars[ cars[cars.columns[y]]==row[y]].count(axis=1).count())
            print("value of x second",x,":",sum)
            if min > sum:
                min = sum
                track = 1
        #Third cluster --> Track =2
        sum = 0
        for y in range(0,4):
            #sum = 0
            if c2[y] != row[y]:
                sum+=(1/cars[ cars[cars.columns[y]]==c2[y]].count(axis=1).count())+(1/cars[ cars[cars.columns[y]]==row[y]].count(axis=1).count())
            print("value of x third",x,":",sum)
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

#print("Cluster Centroids = \n", kmeans.cluster_centers_)

FourCircle['KMeanCluster'] = kmeans.labels_

#for i in range(4):
#    print("Cluster Label = ", i)
#    print(FourCircle.loc[FourCircle['KMeanCluster'] == i])
    
plt.figure(figsize=(8, 6))
plt.scatter(FourCircle['x'], FourCircle['y'], c = FourCircle['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

print("####################### Question 3 C ############################")

import math
import sklearn.neighbors

#for x in range(3,12):
print("After Cheked with different combination of No of Cluster - No of Neighrest neighbor:",
    "Value of Neighrest neighbour :","6")
# Three nearest neighbors
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

evals, evecs = LA.eigh(Lmatrix)

# Series plot of the smallest ten eigenvalues to determine the number of clusters
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()

print("Six eigenvalues Value",evals[0:6,])
# Inspect the values of the selected eigenvectors 
Z = evecs[:,[0,1]]

plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=4, random_state=0).fit(Z)

FourCircle['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(FourCircle['x'], FourCircle['y'], c = FourCircle['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

print("Cluster Centroids = \n", kmeans_spectral.cluster_centers_)
    
