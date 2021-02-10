#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.svm as svm
import itertools


# In[2]:



trainData = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week12_assignment5\\SpiralWithCluster.csv')

y_threshold = trainData['SpectralCluster'].mean()


# In[76]:


# Build Support Vector Machine classifier
xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20191108, max_iter = -1, decision_function_shape  = 'ovr')
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Misclassification = ', 1 - (metrics.accuracy_score(yTrain, y_predictClass)))
trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)


# In[77]:



# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = numpy.linspace(-5, 5)
yy = a * xx - (thisFit.intercept_[0]) / w[1]


# In[78]:


# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[79]:


#quest 2c
# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'], y = subData['y'], c = carray[i], label = i, s = 25)
#plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'x', s = 100)
plt.plot(xx, yy, color = 'black', linestyle = '-')
#plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
#plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
#plt.scatter(cc[:,0], cc[:,1], color = 'black', marker = '+', s = 100)
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


# In[7]:


#########################################Question 2 d ########################################

# Convert to the polar coordinates
trainData['radius'] = numpy.sqrt(trainData['x']**2 + trainData['y']**2)
trainData['theta'] = numpy.arctan2(trainData['y'], trainData['x'])

def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

trainData['theta'] = trainData['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = trainData[['radius','theta']]
yTrain = trainData['SpectralCluster']

print(xTrain.isnull().sum())

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)


# In[8]:




# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i), s = 25)
plt.grid(True)
plt.title('theta-coordinate against the radius-coordinate')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.ylim(-0.5, 6.5)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


# In[50]:


#2e
xTrain = trainData[['radius','theta']]
yTrain = trainData['SpectralCluster']
 #df.loc[df['shield'] > 6]
trainData['Group'] = 2

trainData.loc[(trainData['radius'] < 2.5) & (trainData['theta'] > 3) , ['Group']] = 1
trainData.loc[(trainData['radius'] < 3) & (trainData['theta'] > 5.5) & (trainData['radius'] > 2.5), ['Group']] = 1
trainData.loc[(trainData['radius'] > 2.5) & (trainData['theta'] < 2) , ['Group']] = 3
trainData.loc[(trainData['radius'] > 3.25) & (trainData['theta'] < 3.1) & (trainData['theta'] > 1.9), ['Group']] = 3
# trainData.loc[(trainData['theta'] > 4), ['Group']] = 3
trainData.loc[(trainData['radius'] < 1.5) & (trainData['theta'] > 6), ['Group']] = 0
#trainData


# In[52]:


carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = trainData[trainData['Group'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i), s = 25)
plt.ylim(-0.5, 6.5)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.grid(True)
plt.title('Group wise Clustering Information')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.show()


# In[68]:


df01 = trainData[trainData['Group'].isin([0,1])]
xTrain = df01[['radius','theta']]
yTrain = df01['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print("intercept = ", thisFit.intercept_)
print("coefficient = ", thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx01 = numpy.linspace(1, 5)
yy01 = a * xx01 - (thisFit.intercept_[0]) / w[1]


# In[69]:


df12 = trainData[trainData['Group'].isin([1,2])]
xTrain = df12[['radius','theta']]
yTrain = df12['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print("intercept = ", thisFit.intercept_)
print("coefficient = ", thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx12 = numpy.linspace(1, 5)
yy12 = a * xx12 - (thisFit.intercept_[0]) / w[1]


# In[70]:


df23 = trainData[trainData['Group'].isin([2,3])]
xTrain = df23[['radius','theta']]
yTrain = df23['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print("intercept = ", thisFit.intercept_)
print("coefficient = ", thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx23 = numpy.linspace(1, 5)
yy23 = a * xx23 - (thisFit.intercept_[0]) / w[1]


# In[71]:


carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = trainData[trainData['Group'] == (i)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i), s = 25)

plt.plot(xx01, yy01, color = 'black', linestyle = '-')
plt.plot(xx12, yy12, color = 'red', linestyle = '-')
plt.plot(xx23, yy23, color = 'green', linestyle = '-')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.grid(True)
plt.title('Group wise Clustering Information')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.show()


# In[72]:


h0_xx01 = xx01 * numpy.cos(yy01)
h0_yy01 = xx01 * numpy.sin(yy01)

h0_xx12 = xx12 * numpy.cos(yy12)
h0_yy12 = xx12 * numpy.sin(yy12)

h0_xx23 = xx23 * numpy.cos(yy23)
h0_yy23 = xx23 * numpy.sin(yy23)


# In[83]:


carray = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == (i)]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = (i), s = 25)

plt.plot(h0_xx01, h0_yy01, color = 'black', linestyle = '-',label = 'Hypercurve0')
plt.plot(h0_xx12, h0_yy12, color = 'red', linestyle = '-',label = 'Hypercurve1')
plt.plot(h0_xx23, h0_yy23, color = 'blue', linestyle = '-',label = 'Hypercurve2')
plt.legend(title = 'Cluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.grid(True)
plt.title('Spectral Clustering Information')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.show()


# In[ ]:




