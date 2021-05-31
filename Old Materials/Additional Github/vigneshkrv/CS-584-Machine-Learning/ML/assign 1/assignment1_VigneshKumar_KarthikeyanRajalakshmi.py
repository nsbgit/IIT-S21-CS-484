import numpy as np
import math
from scipy.stats import iqr
i, group, x =np.loadtxt('NormalSample.csv',delimiter=',',skiprows=1,unpack=True)

h = (2*iqr(x))/np.cbrt(x.size)
print(h)
print(max(x))
print(min(x))

# 26
# 36

import math
import matplotlib.pyplot as plt

def densityEstimation(x,h):
    min = 26
    max = 36
    mid = [min+h/2]
    for i in range(1,x.size):
        if(mid[i-1]+h <= max):
            mid.append(mid[i-1]+h)
           
        else: break
        
    p = []
    for i in range(len(mid)):
        sumOfW = 0
        u = []
        w = []
        for j in range(x.size):
            u.append((x[j] - mid[i])/h)            
            if(u[j]>-0.5 and u[j] <= 0.5):
                w.append(1)
            else:
                w.append(0)
            sumOfW = sumOfW + w[j]
        #print(sumOfW)
        p.append(sumOfW / (x.size*h))
        del u
        del w
    return p, mid

h=0.1
p, mid = densityEstimation(x,h)
print(p)
a = plt.figure(1)
plt.bar(mid,p)
a.show
del p

h = 0.5
p, mid = densityEstimation(x,h)
print(p)
a = plt.figure(2)
plt.bar(mid,p)
a.show
del p

h = 1
p, mid = densityEstimation(x,h)
print(p)
a = plt.figure(3)
plt.bar(mid,p)
a.show
del p

h = 2
p, mid = densityEstimation(x,h)
print(p)
a = plt.figure(4)
plt.bar(mid,p)
a.show
del p
del x

# h=1  is captures sufficient data , h=0.1,0.5 gives overall gist 



i, group, x = np.loadtxt('NormalSample.csv',delimiter=',',skiprows=1,unpack=True)
quartiles = np.percentile(x, [25, 50, 75])
print(min(x),quartiles,max(x))

# 1.5 IQR WHISKERS
first = quartiles[0]-1.5*iqr(x)
second=quartiles[2]+1.5*iqr(x)

print(first)
print(second)

group_1=[];
group_2=[];
for a,b in enumerate(group):
    if b==0.0:
        group_1.append(x[a])
    else:
        group_2.append(x[a])


quartiles = np.percentile(group_1, [25, 50, 75])
print(min(group_1),quartiles,max(group_1))

# 1.5 IQR WHISKERS
first = quartiles[0]-1.5*iqr(group_1)
second=quartiles[2]+1.5*iqr(group_1)
print(first)
print(second)

quartiles = np.percentile(group_2, [25, 50, 75])
print(min(group_2),quartiles,max(group_2))

# 1.5 IQR WHISKERS
first = quartiles[0]-1.5*iqr(group_2)
second=quartiles[2]+1.5*iqr(group_2)
print(first)
print(second)

import matplotlib.pyplot as plt
box_fig = plt.figure(5)
y=plt.boxplot(x)
box_fig.show()
[i.get_ydata() for i in y['whiskers']] # boxplot values similar to 1.5 *IQR values

box_fig=plt.figure(6)
y2=plt.boxplot([x,group_1,group_2],positions=[1,2,3])
box_fig.show()
[i.get_ydata() for i in y2['whiskers']]

import pandas as pd
fraudulent = pd.read_csv('Fraud.csv')
# print(fraudulent)
print(" Fraud % = ", round((fraudulent.iloc[:,1].sum()/fraudulent.iloc[:,1].size)*100,4))


fraud_grouping = fraudulent.groupby('FRAUD')
fraud_groups = [fraud_grouping.get_group(G) for G in fraud_grouping.groups]

totalSpend=plt.figure(7)
plt.boxplot([fraud_groups[0].iloc[:,2],fraud_groups[1].iloc[:,2]],positions=[1,2],vert=False)
plt.title("Total Spend")
totalSpend.show()

doctor=plt.figure(8)
plt.boxplot([fraud_groups[0].iloc[:,3],fraud_groups[1].iloc[:,3]],positions=[1,2],vert=False)
plt.title("Doctor visitis")
doctor.show()

Num_claims=plt.figure(9)
plt.boxplot([fraud_groups[0].iloc[:,4],fraud_groups[1].iloc[:,4]],positions=[1,2],vert=False)
plt.title("Num claims")
Num_claims.show()

MmemberDuration=plt.figure(10)
plt.boxplot([fraud_groups[0].iloc[:,5],fraud_groups[1].iloc[:,5]],positions=[1,2],vert=False)
plt.title("MmemberDuration ")
MmemberDuration.show()

OPTOM_PRESC=plt.figure(11)
plt.boxplot([fraud_groups[0].iloc[:,6],fraud_groups[1].iloc[:,6]],positions=[1,2],vert=False)
plt.title("OPTOM_PRESC ")
OPTOM_PRESC.show()

NUM_MEMBERS=plt.figure(12)
plt.boxplot([fraud_groups[0].iloc[:,7],fraud_groups[1].iloc[:,7]],positions=[1,2],vert=False)
plt.title("NUM_MEMBERS")
NUM_MEMBERS.show()

fraud_val = fraudulent.values
x = np.delete(fraud_val,[0,1],1)
print(x)
# print("Number of Dimensions = ", x.ndim)

# Transpose multiplication 
xtx = np.matmul(x.transpose(),x)
print(xtx)

from numpy import linalg as LA
import scipy
from scipy import linalg as LA2
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier

eigen_val, eigen_vec = LA.eigh(xtx)
print(eigen_val)
print(eigen_vec)

# transformation matrix
transformationMatrix = eigen_vec.dot(LA.inv(np.sqrt(np.diagflat(eigen_val))));
print("Transformation Matrix = \n", transformationMatrix)

transformedX = np.matmul(x,transformationMatrix)
print("The Transformed x = \n", transformedX)

xtx = np.matmul(transformedX.transpose(),transformedX);
print("Identity Matrix = \n", np.round(xtx))

orthonormalizeX = LA2.orth(x)
print("orthonormalize x = \n", orthonormalizeX)
crossCheck = orthonormalizeX.transpose().dot(orthonormalizeX)
print("Identity Matrix again = \n", np.round(crossCheck))

nearestNeigh=kNN(n_neighbors=5,algorithm='brute',metric='euclidean')
training=transformedX
neighbors=nearestNeigh.fit(training)
# edges, vertices = neighbors.kneighbors(training)
sampleData=[[7500,15,3,127,2,2]]
transformingSample=np.matmul(sampleData,transformationMatrix)
neighborsNearMe = neighbors.kneighbors(transformingSample, return_distance = False)
print(neighborsNearMe)

testing=fraudulent[['FRAUD']]
near = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
neig=near.fit(training,testing)
result=neig.predict(training)
probab=neig.predict_proba(transformingSample)
accuracy = neig.score(training, testing)
print(accuracy)
print(fraudulent.iloc[588,:])
print("----------------------------------------------------------------")
print(fraudulent.iloc[2897,:])
print("----------------------------------------------------------------")
print(fraudulent.iloc[1199,:])
print("----------------------------------------------------------------")
print(fraudulent.iloc[1246,:])
print("----------------------------------------------------------------")
print(fraudulent.iloc[886,:])