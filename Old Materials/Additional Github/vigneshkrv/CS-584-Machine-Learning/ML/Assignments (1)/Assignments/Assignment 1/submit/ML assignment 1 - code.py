
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import scipy
from scipy import linalg as LA2
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import iqr
x, group = np.loadtxt('NormalSample.csv', delimiter = ',', unpack = True)
print(x)
print(iqr(x))
h = (2*iqr(x))/np.cbrt(x.size)
print("h = ",h)


# In[11]:


u = np.log10(h)
v = np.sign(u) * math.ceil(abs(u))
h = np.power(10,v)
print(h)


# In[12]:


N  = x.size
h = 0.5


        
def find_density_estimates(x,h):
    min = 45
    max = 55
    mid_points = [min+h/2]
    for i in range(1,x.size):
        if(mid_points[i-1]+h <= max):
            mid_points.append(mid_points[i-1]+h)
            #print(mid_points[i])
        else: break
        
    p = []
    for i in range(len(mid_points)):
        w_sum = 0
        u = []
        w = []
        for j in range(x.size):
            u.append((x[j] - mid_points[i])/h)            
            if(u[j]>-0.5 and u[j] <= 0.5):
                w.append(1)
            else:
                w.append(0)
            w_sum = w_sum + w[j]
        #print(w_sum)
        p.append(w_sum / (x.size*h))
        del u
        del w
    return p, mid_points

p, mid_points = find_density_estimates(x,h)
print(p)
a = plt.figure(1)
plt.bar(mid_points,p)
a.show
del p

h = 1
p, mid_points = find_density_estimates(x,h)
print(p)
a = plt.figure(2)
plt.bar(mid_points,p)
a.show
del p

h = 2
p, mid_points = find_density_estimates(x,h)
print(p)
a = plt.figure(3)
plt.bar(mid_points,p)
a.show
del p
del x


# In[13]:


data = pd.read_csv("NormalSample_labelled.csv")
col_x = data.iloc[:,0]
# calculate quartiles
quartiles = np.percentile(col_x, [25, 50, 75])
# calculate min/max
data_min, data_max = col_x.min(), col_x.max()
print(data_min,quartiles,data_max)

grpBy = data.groupby('Group')
groups = [grpBy.get_group(g) for g in grpBy.groups]

g0_max = groups[0].iloc[:,0].max()
g0_min = groups[0].iloc[:,0].min()
g0_25, g0_50, g0_75 = np.percentile(groups[0].iloc[:,0], [25, 50, 75])
print(g0_min,g0_25,g0_50,g0_75,g0_max)

g1_max = groups[1].iloc[:,0].max()
g1_min = groups[1].iloc[:,0].min()
g1_25, g1_50, g1_75 = np.percentile(groups[1].iloc[:,0], [25, 50, 75])
print(g1_min,g1_25,g1_50,g1_75,g1_max)

box_fig = plt.figure(4)
plt.boxplot(col_x)
box_fig.show()

box_plot = plt.figure(5)
plt.boxplot([col_x, groups[0].iloc[:,0], groups[1].iloc[:,0]],positions = [1,2,3])
box_plot.show()


# In[14]:


df = pd.read_csv('Fraud.csv')
# Input the matrix X
print(" Fraud % = ", (df.iloc[:,1].sum()/df.iloc[:,1].size)*100)
fraud_grpBy = df.groupby('FRAUD')
fraud_grps = [fraud_grpBy.get_group(g) for g in fraud_grpBy.groups]

box_plot = plt.figure(6)
plt.boxplot([fraud_grps[0].iloc[:,2],fraud_grps[1].iloc[:,2]],positions = [1,2],vert = False)
plt.title('TOTAL_SPEND')
box_plot.show()

box_plot = plt.figure(7)
plt.boxplot([fraud_grps[0].iloc[:,3],fraud_grps[1].iloc[:,4]],positions = [1,2],vert = False)
plt.title('DOCTOR_VISITS')
box_plot.show()

box_plot = plt.figure(8)
plt.boxplot([fraud_grps[0].iloc[:,4],fraud_grps[1].iloc[:,4]],positions = [1,2],vert = False)
plt.title('NUM_CLAIMS')
box_plot.show()

box_plot = plt.figure(9)
plt.boxplot([fraud_grps[0].iloc[:,5],fraud_grps[1].iloc[:,5]],positions = [1,2],vert = False)
plt.title('MEMBER_DURATION')
box_plot.show()

box_plot = plt.figure(10)
plt.boxplot([fraud_grps[0].iloc[:,6],fraud_grps[1].iloc[:,6]],positions = [1,2],vert = False)
plt.title('OPTOM_PRESC')
box_plot.show()

box_plot = plt.figure(11)
plt.boxplot([fraud_grps[0].iloc[:,7],fraud_grps[1].iloc[:,7]],positions = [1,2],vert = False)
plt.title('NUM_MEMBERS')
box_plot.show()

df_vals = df.values
x = np.delete(df_vals,[0,1],1)

print("Input Matrix = \n", x)

print("Number of Dimensions = ", x.ndim)

print("Number of Rows = ", np.size(x,0))
print("Number of Columns = ", np.size(x,1))

#xtx = x.transpose() * x
xtx = np.matmul(x.transpose(),x)
print("t(x) * x = \n", xtx)

# Eigenvalue decomposition
evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
transf = evecs.dot(LA.inv(np.sqrt(np.diagflat(evals))));
print("Transformation Matrix = \n", transf)

# Here is the transformed X
#transf_x = x * transf;
transf_x = np.matmul(x,transf)
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
#xtx = transf_x.transpose() * transf_x;
xtx = np.matmul(transf_x.transpose(),transf_x);
print("Expect an Identity Matrix = \n", np.round(xtx))

# Orthonormalize using the orth function 

orthx = LA2.orth(x)
print("The orthonormalize x = \n", orthx)

# Check columns of the ORTH function
check = orthx.transpose().dot(orthx)
print("Also Expect an Identity Matrix = \n", np.round(check))


# In[15]:


kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')

#trainData = df[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]
#trainData.describe()
trainData = transf_x
# Build nearest neighbors
nbrs = kNNSpec.fit(trainData)
distances, indices = nbrs.kneighbors(trainData)

# Find the nearest neighbors of these focal observations       
#focal = [[7500, 15, 3, 127, 2, 2]]     # Mercedes-Benz_271
sample = [[7500, 15, 3, 127, 2, 2]]
#sample.reshape(1,-1)
transf_samp = np.matmul(sample,transf)

myNeighbors = nbrs.kneighbors(transf_samp, return_distance = False)
print("My Neighbors = \n", myNeighbors)

# Perform classification
target = df[['FRAUD']]


neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)

# See the classification result
class_result = nbrs.predict(trainData)
#print(class_result)

# See the classification probabilities
#class_prob = nbrs.predict_proba(trainData)
class_prob = nbrs.predict_proba(transf_samp)
print(class_prob)

accuracy = nbrs.score(trainData, target)
print(accuracy)
print(df.iloc[588,:])
print(df.iloc[2897,:])
print(df.iloc[1199,:])
print(df.iloc[1246,:])
print(df.iloc[886,:])

