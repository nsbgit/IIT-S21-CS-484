############################## Question 1######################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\Week2\\NormalSample.csv',
                       delimiter=',')

x = data.iloc[:,2].values
#print(x)
Quat = np.percentile(x, [25,50,75])
#print(Quat)
IQR = Quat[2] - Quat[0]
#print(IQR)
N = len(x)
bin_width = 2 * IQR*(N ** (-1/3))

print("According to Izenman (1991) method, the recomemded bin width for Histogram x is ", bin_width)

print("The minimum and maximum value of the field x are", np.min(x) , "and" , np.max(x),"respectively")

a = math.floor(np.min(x))
b = math.ceil(np.max(x))

print("The Values of a and b are", a, "and ", b, "respectively")

h= 0.1
m = a + h/2
m_values = []
p = []

while (1):
    m_values.append(m)
    w_u = []
    
    u = (x-m)/h
    for i in u:
        if i> -1/2 and i < 1/2:
            w_u.append(1)
        else:
            w_u.append(0)
    p.append(sum(w_u)/(N * h))
    if m > b:
        break
    else:
        m = m + h

print(list(zip(m_values,p)))


num_bins = int((b-a)/h)

plt.hist(x, bins= num_bins)
plt.title("Histogram of Normal Sample for h = 0.1")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(axis="x")
plt.show()


###################################### E part##################################

h= 0.5
m = a + h/2
m_values = []
p = []

while (1):
    m_values.append(m)
    w_u = []
    
    u = (x-m)/h
    for i in u:
        if i> -1/2 and i < 1/2:
            w_u.append(1)
        else:
            w_u.append(0)
    p.append(sum(w_u)/(N * h))
    if m > b:
        break
    else:
        m = m + h
#print(len(p))
#print(len(m_values))
print(list(zip(m_values,p)))

num_bins = int((b-a)/h)

plt.hist(x, bins= num_bins)
plt.title("Histogram of Normal Sample for h = 0.5")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(axis="x")
plt.show()

###################################### F part##################################

h= 1
m = a + h/2
m_values = []
p = []

while (1):
    m_values.append(m)
    w_u = []
    
    u = (x-m)/h
    for i in u:
        if i> -1/2 and i < 1/2:
            w_u.append(1)
        else:
            w_u.append(0)
    p.append(sum(w_u)/(N * h))
    if m > b:
        break
    else:
        m = m + h
#print(len(p))
#print(len(m_values))
print(list(zip(m_values,p)))

num_bins = int((b-a)/h)

plt.hist(x, bins= num_bins)
plt.title("Histogram of Normal Sample for h = 1")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(axis="x")
plt.show()

###################################### G part##################################

h= 2
m = a + h/2
m_values = []
p = []
d = dict()
counter = 0
while (1):
    m_values.append(m)
    w_u = []
    
    u = (x-m)/h
    #print("value of u", u)
    for i in u:
        if i> -1/2 and i < 1/2:
            w_u.append(1)
        else:
            w_u.append(0)
   # print(m_values[counter], p[counter])
    
    
    p.append(sum(w_u)/(N * h))
    print(m_values[counter],':', p[counter])
    counter = counter + 1
    

    if m > b:
        break
    else:
        d[m] = p
        m = m + h

print(list(zip(m_values,p)))
print("dictionary")
print(p[0])


num_bins = int((b-a)/h)

plt.hist(x, bins= num_bins)
plt.title("Histogram of Normal Sample for h = 2")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(axis="x")
plt.show()

##############################Question 2 ##############################

#data['x'].describe()
lower_Whisker  = np.subtract(Quat[0],(1.5 * IQR ))
upper_Whisker  = np.add(Quat[2], (1.5 * IQR))
############################ A part #####################################
print("The five-number summary:")
print("The minimum, the first quartile, the median, the third quartile, and the maximum are",np.min(x), ",", Quat[0], ",", Quat[1], ",", Quat[2], "and", np.max(x))
print("Lower Whisker and Upper Whisker values are ", lower_Whisker, "and ", upper_Whisker, "respectively")

############################ B part #####################################

data['x'][data['group']==0].describe()

data['x'][data['group']==1].describe()

########################### C part #######################################
boxPlot_x = [x]
plt.boxplot(boxPlot_x, vert=0, patch_artist=True)
plt.title("Box plot of x")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(axis="x")
plt.show()

########################### D part #######################################
p = data[data['group']==0]
q = data[data['group']==1]
box_data = [data['x'], p['x'], q['x']]
plt.boxplot(box_data, vert=0,patch_artist= True)
plt.title("Box plot of x for each category of Group")
plt.xlabel("x")
plt.grid(axis="y")
plt.show()

########################## Question 3####################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA

fraud = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\Week2\\Fraud.csv',
                       delimiter=',')

#print(fraud)
################################# A part ########################################
print(np.around(fraud['FRAUD'].value_counts()[1]/fraud.shape[0], 6) * 100) #percentage

################################# B part ########################################
fradulent = fraud[fraud['FRAUD'] == 1]  
nonFradulent = fraud[fraud['FRAUD'] != 1]
var_Array = fraud.keys().tolist()
var_Array.remove('CASE_ID')
var_Array.remove('FRAUD')

for i in range(0, len(var_Array)):
  boxPlotData =[nonFradulent[var_Array[i]], fradulent[var_Array[i]], ];
  plt.boxplot(boxPlotData, vert=0, labels=[0,1], patch_artist= True);
  plt.title(var_Array[i])
  plt.show();

################################# C part ########################################
x = np.matrix(fraud.drop(['CASE_ID','FRAUD'], axis=1))
xtx = x.transpose() * x
evalues, evects = LA.eigh(xtx)

#print(evalues)
#print(evects)

transf = evects * LA.inv(np.sqrt(np.diagflat(evalues)));
print("Transformation Matrix = \n", transf)

transf_x = x * transf;
print("The Transformed x = \n", transf_x)
xtx = transf_x.transpose() * transf_x;
print("Identity Matrix = \n", xtx)
xtx.shape

from scipy import linalg as LA2

orthx = LA2.orth(x)
print("The orthonormalize x = \n", orthx)


check = orthx.transpose().dot(orthx)
print("Identity Matrix = \n", check)

################################# D part ########################################
from sklearn.neighbors import KNeighborsClassifier

trainData = fraud.drop(['CASE_ID','FRAUD'], axis=1)
target = fraud['FRAUD'];

neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)
accu = nbrs.score(x, target)
print(accu)

################################# E part ########################################
from sklearn.neighbors import NearestNeighbors as knn
focal = [7500,15,3,127,2,2]
transf_focal = focal * transf;
neigh_t = knn(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs_t = neigh_t.fit(transf_x)

myNeighbors_t = nbrs_t.kneighbors(transf_focal, return_distance = False)
print("My Neighbors = ", myNeighbors_t)


############################### F part #########################################

print("Predicted Probability of Fraudulent ",nbrs.predict(transf_x))
