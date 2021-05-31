import pandas as pd
import numpy as np
from kmodes.kmodes import KModes

df = pd.read_csv("cars.csv")
type_list = df['Type']
origin_list = df['Origin']
drivetrain_list = df['DriveTrain']
cylinders_list = df['Cylinders']

print("Question 1")
unique_elements, counts_elements = np.unique(type_list, return_counts=True)
elements = np.asarray(unique_elements)
count = np.asarray(counts_elements)
print("Frequencies of the categorical feature Type: ")
#print(unique_elements,counts_elements)

for i in range(len(elements)):
    print(elements[i],":",count[i])

print("\nQuestion 2")
unique_elements, counts_elements = np.unique(drivetrain_list, return_counts=True)
elements = np.asarray(unique_elements)
count = np.asarray(counts_elements)
print("Frequencies of the categorical feature DriveTrain: ")
for i in range(len(elements)):
    print(elements[i],":",count[i])

print("\nQuestion 3")
unique_elements, counts_elements = np.unique(origin_list, return_counts=True)
elements = np.asarray(unique_elements)
count = np.asarray(counts_elements)
print("Frequencies of the categorical feature Origin: ")
for i in range(len(elements)):
    print(elements[i],":",count[i])

print("\nDistance between Origin=Asia and Origin=Europe: ")
distance = 1/count[0]+1/count[1]
print(distance)

print("\nQuestion 4")
print("Frequencies of the categorical feature Cylinders: ")
cylinder_count = cylinders_list.value_counts()
cylinder_miss = cylinders_list.isnull()
print(cylinder_count)
counts = 0

for i in range(0,len(cylinder_miss)):
    if cylinder_miss[i] == True:
        counts+=1

print("Missing Values: ", counts)
value = cylinder_count[[5.0]]
distance = 1/value[5.0]+1/counts
print("Distance between Cylinders=5 and Cylinders=Missing: ")
print(distance)


data_kmodes = df[['Type','Origin','DriveTrain','Cylinders']]
data_kmodes = data_kmodes.fillna(-1)
km = KModes(n_clusters=3, init='Huang', n_init=10)
clusters = km.fit_predict(data_kmodes)

observation_1=list(clusters).count(0)
observation_2=list(clusters).count(1)
observation_3=list(clusters).count(2)
observation_list = [observation_1,observation_2,observation_3]
print("\nQuestion 5")
print("Number of Observation in in three clusters: ")
print("1.",observation_1)
print("2.",observation_2)
print("3.",observation_3)
print("\nCentroids: ")
print(km.cluster_centroids_)

print("\nQuestion 6")

data_kmodes['Cluster_number']=pd.Series(clusters)
print(pd.crosstab(index = data_kmodes["Origin"], columns = data_kmodes["Cluster_number"]))