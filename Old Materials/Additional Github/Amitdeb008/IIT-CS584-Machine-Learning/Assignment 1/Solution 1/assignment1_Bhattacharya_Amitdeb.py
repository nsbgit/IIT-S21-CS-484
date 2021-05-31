#!/usr/bin/env python
# coding: utf-8

# In[88]:


#importing required python libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[90]:


#importing given datasets
fraud_dataset = pd.read_csv("C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 1\\Fraud.csv")
sample_dataset = pd.read_csv("C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 1\\NormalSample.csv")


# In[91]:


#displaying top few rows of fraud dataset
print("Fraud Dataset:")
fraud_dataset.head()


# In[86]:


#displaying top few rows of sample dataset
print("NormalSample Dataset:")
sample_dataset.head()


# In[87]:


#checking the total length of the sample_dataset
print("Length of sample_dataset: {}".format(len(sample_dataset['x'])))


# In[92]:


iqr = sample_dataset['x'].describe()[6] - sample_dataset['x'].describe()[4]
print("Interquartile range of X values: {}".format(iqr))


# In[93]:


h = 2*iqr*(np.power(len(sample_dataset['x']),-1/3))
print("According to Lzenmanh the recomended bin-width for histogram of x: {}".format(h))


# In[94]:


u =np.log10(h)
v =np.sign(u)*np.ceil(np.abs(u))
bin_width = np.power(10,v)
print("Nice h value in Practice: {}".format(bin_width))


# In[95]:


min_value = sample_dataset['x'].min()
print("Minimum value of field x: {}".format(min_value))


# In[96]:


max_value = sample_dataset['x'].max()
print("Maximum Value of field x: {}".format(max_value))


# In[97]:


a = np.floor(min_value)
print("Largest integer less than the minimum value of the field x: {}".format(a))


# In[98]:


b = np.ceil(max_value)
print("The smallest integer greater than the maximum value of the field x : {}".format(b))


# In[99]:


h = 0.1
minimum = 26
maximum = 36


# In[107]:


#plotting a histogram with = 0.1
mid_point = [e + 0.1/2 for e in np.arange(26,36,0.1)][:-1]
density = plt.hist(x=sample_dataset['x'],bins=np.arange(26,36,0.1),density=True)
plt.xlabel("<---- X values ---->")
plt.ylabel("<---- Density ---->")
plt.title("Histogram with bin-width: 0.1")
plt.show()


# In[108]:


#storing the co-ordinates of density estimator
co_ord = []
for i in range(len(mid_point)):
    co_ord.append((np.round(mid_point[i],2),np.round(density[0][i],5)))


# In[109]:


#displaying first 10 co-ordinates
print("Displaying First 20 Co-ordinates of Density estimator: ")
co_ord[:20]


# In[110]:


#plotting a histogram with bin-width 0.5
h = 0.5
mid_point = [e + h/2 for e in np.arange(26,36,h)][:-1]
density = plt.hist(x=sample_dataset['x'],bins=np.arange(26,36,h),density=True)
plt.xlabel("<---- X values ---->")
plt.ylabel("<---- Density ---->")
plt.title("Histogram with bin-width:{}".format(h))
plt.show()


# In[111]:


#storing the co-ordinates of density estimator
co_ord = []
for i in range(len(mid_point)):
    co_ord.append((np.round(mid_point[i],2),np.round(density[0][i],5)))


# In[112]:


#displaying first few co-ordinates
print("Displaying First 20 Co-ordinates of Density estimator:")
co_ord[:20]


# In[113]:


#plotting a histogram with bin-width 1.0
h = 1.0
mid_point = [e + h/2 for e in np.arange(26,36,h)][:-1]
density = plt.hist(x=sample_dataset['x'],bins=np.arange(26,36,h),density=True)
plt.xlabel("<---- X values ---->")
plt.ylabel("<---- Density ---->")
plt.title("Histogram with bin-width:{}".format(h))
plt.show()


# In[114]:


#storing the co-ordinates of density estimator
co_ord = []
for i in range(len(mid_point)):
    co_ord.append((np.round(mid_point[i],2),np.round(density[0][i],5)))


# In[115]:


#displaying first few co-ordinates
print("Displaying First few Co-ordinates of Density estimator:" )
co_ord[:10]


# In[117]:


#plotting a histogram with bin-width 2.0
h = 2.0
mid_point = [e + h/2 for e in np.arange(26,36,h)][:-1]
density = plt.hist(x=sample_dataset['x'],bins=np.arange(26,36,h),density=True)
plt.xlabel("<---- X values ---->")
plt.ylabel("<---- Density ---->")
plt.title("Histogram with bin-width:{}".format(h))
plt.show()


# In[118]:


#storing the co-ordinates of density estimator
co_ord = []
for i in range(len(mid_point)):
    co_ord.append((np.round(mid_point[i],2),np.round(density[0][i],5)))


# In[120]:


#displaying first few co-ordinates
print("Displaying First few Co-ordinates of Density estimator: ")
co_ord


# In[163]:


#five-number summary of x
print("Five number summary:\n{}".format(sample_dataset['x'].describe()))


# In[164]:


#Calculating the 1.5 IQR Whisker values
iqr = sample_dataset['x'].describe()[6] - sample_dataset['x'].describe()[4]
upper_limit = sample_dataset['x'].describe()[6] + 1.5*iqr
lower_limit = sample_dataset['x'].describe()[4] - 1.5*iqr


# In[165]:


print("Value of 1.5 IQR upperlimit: {} and lowerlimit: {}".format(upper_limit,lower_limit))


# In[166]:


#five-number summary of x for each category of the group
sample_dataset.groupby('group').describe()['x']


# In[167]:


#Calculating Q1,Q3,IQR of group 0
q1_0 = sample_dataset.groupby('group').describe()['x']['25%'][0]
q3_0 = sample_dataset.groupby('group').describe()['x']['75%'][0]
iqr_0 = round(q3_0 - q1_0,2)
iqr_0


# In[168]:


upper_limit_group0 = q3_0 + 1.5*iqr_0
lower_limit_group0 = round(q1_0 - 1.5*iqr_0,2)
print("Values of the 1.5 IQR whiskers for group 0 Upperlimit: {} and Lowerlimit: {}"
.format(upper_limit_group0,lower_limit_group0))


# In[169]:


#Calculating Q1,Q3,IQR of group 1
q1_1 = sample_dataset.groupby('group').describe()['x']['25%'][1]
q3_1 = sample_dataset.groupby('group').describe()['x']['75%'][1]
iqr_1 = round(q3_1 - q1_1,2)
iqr_1


# In[170]:


upper_limit_group1 = round(q3_1 + 1.5*iqr_1,2)
lower_limit_group1 = round(q1_1 - 1.5*iqr_1,2)
print("Values of the 1.5 IQR whiskers for group 1 Upperlimit: {} and LowerLimit:{}"
.format(upper_limit_group1,lower_limit_group1))


# In[171]:


#Code to draw boxplot
sns.set(style="whitegrid")
sns.boxplot(sample_dataset['x'],orient="v")
plt.show()


# In[172]:


upper_limit = sample_dataset['x'].describe()[6] + 1.5*iqr
lower_limit = sample_dataset['x'].describe()[4] - 1.5*iqr


# In[173]:


#Checking the upper limit and lower limit values
(upper_limit,lower_limit)


# In[174]:


#Extracting x-values of group 0
grouped0 = sample_dataset.groupby('group').get_group(0)


# In[175]:


#Extracting x-values of group 1
grouped1 = sample_dataset.groupby('group').get_group(1)


# In[176]:


#Renaming x cloumn of group0 to x_group0
grouped0 = grouped0.rename(columns={'x':'x_group0'})


# In[177]:


#Renaming x cloumn of group1 to x_group1
grouped1 = grouped1.rename(columns={'x':'x_group1'})


# In[178]:


#Extracting x-values of group 0 to a list
group0 = grouped0['x_group0'].tolist()


# In[179]:


#Extracting x-values of group 1 to a list
group1 = grouped1['x_group1'].tolist()


# In[180]:


#Extracting x-values of both groups from original dataframe to a list
xvalues = sample_dataset['x'].tolist()


# In[182]:


#Plotting the box plot of Group-0 x-values, Group-1 x-values and x-values
plt.boxplot([group0,group1,xvalues],labels=['group0','group1','x'],sym='k.')
plt.title("Boxplot of Group-0 x-values, Group-1 x-values and x-values")
plt.show()


# In[219]:


#displaying top few rows of dataset
fraud_dataset.head()


# In[220]:


#info of dataframe
fraud_dataset.info()


# In[221]:


print("The fraudulent investigations percentage: {}".format(round(np.mean(fraud_dataset['FRAUD'])*100,4)))


# In[222]:


#Fraud vs Total_spend
sns.boxplot(x="TOTAL_SPEND",y="FRAUD",orient='h',data=fraud_dataset)
plt.show()


# In[223]:


#Fraud VS Doctor visits
sns.boxplot(x="DOCTOR_VISITS",y="FRAUD",orient='h',data=fraud_dataset)
plt.show()


# In[224]:


#Fraud vs Num_claims
sns.boxplot(x="NUM_CLAIMS",y="FRAUD",orient='h',data=fraud_dataset)
plt.show()


# In[225]:


#Fraud vs Member_Duration
sns.boxplot(x="MEMBER_DURATION",y="FRAUD",orient='h',data=fraud_dataset)
plt.show()


# In[226]:


#Fraud vs Optom_Presc
sns.boxplot(x="OPTOM_PRESC",y="FRAUD",orient='h',data=fraud_dataset)
plt.show()


# In[227]:


#Fraud vs Num_Memebers
sns.boxplot(x="NUM_MEMBERS",y="FRAUD",orient='h',data=fraud_dataset)
plt.show()


# In[228]:


#displaying top 5 rows of dataset
fraud_dataset.head()


# In[245]:


# Orthonormalizing interval variables from the dataframe
'''
- Ignoring case ID and fraud because fraud is our target variable and cases ID is not a interval variable.
- Extracting the fields from total spend to num_member converting it into a matrix and storing them in x.
'''
x = np.matrix(fraud_dataset.iloc[:,2:].values)


# In[246]:


#checking the shape of x value
x.shape


# In[247]:


#multiplying x transpose with x
xtx = x.transpose() * x


# In[248]:


#eigen value decomposition
eigen_value, eigen_vector = np.linalg.eigh(xtx)


# In[250]:


print("Eigen values of x = \n", eigen_value)
print("Eigen vectors of x = \n",eigen_vector)


# In[251]:


#checking if all interval variable eigen values are greater than 1
eigen_value > 1


# In[253]:


#Transformation matrix
transformation = eigen_vector * np.linalg.inv(np.sqrt(np.diagflat(eigen_value)));
print("Transformation Matrix = \n", transformation)


# In[255]:


# Here is the transformed X
transformation_x = x * transformation;
print("The Transformed x = \n", transformation_x)


# In[260]:


# Proof to check resulting varaibles are othonormal
xtx = transformation_x.transpose() * transformation_x;
print("Expect an Identity Matrix = \n", xtx)


# In[261]:


#Checking whether 1.00000000e+00 is equal to 1
1.00000000e+00 == 1


# In[267]:


#Loading the NearestNeighbors library
from sklearn.neighbors import KNeighborsClassifier


# In[268]:


#Extracting target values from fraud dataframe
target = fraud_dataset.iloc[:,1]


# In[269]:


#implementing the Nearest neighbours algorithm by using 5 neighbours 
neigh = KNeighborsClassifier(n_neighbors=5,algorithm = 'brute', metric = 'euclidean')


# In[270]:


#training the model
nbrs = neigh.fit(transformation_x,target)


# In[276]:


#result using score function
print("Score function result: {}".format(nbrs.score(transformation_x,target)))


# In[283]:


#Transforming i/p obsevations as scaling is required for accuracy for KNN algorithm.
focal = [[7500, 15, 3, 127, 2, 2]]
trans_focal = focal * transformation


# In[284]:


test_set = nbrs.kneighbors(trans_focal,n_neighbors = 5)
print("Distance values: {} \nNearest Neighbours: {}".format(test_set[0],test_set[1]))


# In[291]:


#Prediction label of test data
print("Predicted label of KNN for Test data:{}".format(nbrs.predict(trans_focal)))


# In[294]:


#probability of testdata being class 0 or class 1
print("Checking the probability values of test set: {}".format(nbrs.predict_proba(trans_focal)))

