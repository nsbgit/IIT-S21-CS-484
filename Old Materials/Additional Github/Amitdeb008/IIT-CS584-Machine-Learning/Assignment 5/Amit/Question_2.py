#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import Libraries
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.svm as svm


# In[9]:


train_Data = pandas.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 5\\SpiralWithCluster.csv')
y_thrsld = train_Data['SpectralCluster'].mean()


# Question 2

# a) (5 points) What is the equation of the separating hyperplane?  Please state the coefficients up to seven decimal places.

# In[10]:


x_Train = train_Data[['x','y']]
y_Train = train_Data['SpectralCluster']

svm_model = svm.SVC(kernel = 'linear', random_state = 20191108, max_iter = -1, 
                    decision_function_shape  = 'ovr')
thisFit = svm_model.fit(x_Train, y_Train)
y_predictClass = thisFit.predict(x_Train)

train_Data['_PredictedClass_'] = y_predictClass

svm_Mean = train_Data.groupby('_PredictedClass_').mean()
print(svm_Mean)
print(" ")
print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = numpy.linspace(-3, 3)
yy = a * xx - (thisFit.intercept_[0]) / w[1]


# b) (5 points) What is the misclassification rate?

# In[5]:


misclassification  = 1 - metrics.accuracy_score(y_Train, y_predictClass)
print("Misclassification:", misclassification)


# c)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[6]:


b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = train_Data[train_Data['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'x', s = 100)
plt.plot(xx, yy, color = 'black', linestyle = '-')
plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


# d)	(10 points) Please express the data as polar coordinates.  Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster variable (0 = Red and 1 = Blue).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[11]:


train_Data['radius'] = numpy.sqrt(train_Data['x']**2 + train_Data['y']**2)
train_Data['theta'] = numpy.arctan2(train_Data['y'], train_Data['x'])

def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

train_Data['theta'] = train_Data['theta'].apply(customArcTan)

x_Train = train_Data[['radius','theta']]
y_Train = train_Data['SpectralCluster']

print(x_Train.isnull().sum())

svm_model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_model.fit(x_Train, y_Train) 
y_predictClass = thisFit.predict(x_Train)

print('Mean Accuracy = ', metrics.accuracy_score(y_Train, y_predictClass))
train_Data['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

xx = numpy.linspace(0, 6)
yy = numpy.zeros((len(xx),3))
for j in range(1):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]


carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = train_Data[train_Data['_PredictedClass_'] == (i+1)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i+1), s = 25)
plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Support Vector Machines on Three Segments')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.ylim(-0.5, 6.5)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


# e)   (10 points) You should expect to see three distinct strips of points and a lone point.  Since the
# SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new
# target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of
# the chart in (d), values 1, 2,and 3 for the next three strips of points. Please plot the theta-coordinate against
# the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red,
# 1 = Blue, 2 = Green, 3 = Black).  To obtain the full credits, you should properly label the axes, the legend,
# and the chart title.  Also, grid lines should be added to the axes.

# In[25]:


group = numpy.zeros(train_Data.shape[0])

for index, row in train_Data.iterrows():
    if row['radius'] < 1.5 and row['theta'] > 6:
        group[index] = 0
    elif row['radius'] < 2.5 and row['theta'] > 3:
        group[index] = 1
    elif 2.5 < row['radius'] < 3 and row['theta'] > 5.5:
        group[index] = 1
    elif row['radius'] < 2.5 and row['theta'] < 3:
        group[index] = 2
    elif 3 < row['radius'] < 4 and 3.5 < row['theta'] < 6.5:
        group[index] = 2
    elif 2.5 < row['radius'] < 3 and 2 < row['theta'] < 4:
        group[index] = 2
    elif 2.5 < row['radius'] < 3.5 and row['theta'] < 2.25:
        group[index] = 3
    elif 3.55 < row['radius'] and row['theta'] < 3.25:
        group[index] = 3

train_Data['group'] = group

color_array = ['red', 'blue', 'green', 'black']
for i in range(4):
    x_y = train_Data[train_Data['group'] == i]
    plt.scatter(x=x_y['radius'], y=x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius ----->')
plt.ylabel('Theta ----->')

plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()
print('Support Vector Machines on Four Segments')


# f) (10 points) Since the graph in (e) has four clearly separable and neighboring segments, we will apply the
# Support Vector Machine algorithm in a different way.  Instead of applying SVM once on a multi-class target
# variable, you will SVM three times, each on a binary target variable. SVM 0: Group 0 versus Group 1 SVM 1: Group 1
# versus Group 2 SVM 2: Group 2 versus Group 3 Please give the equations of the three hyperplanes.

# In[34]:


# build SVM 0: Group 0 versus Group 1
svm_1 = svm.SVC(kernel="linear", random_state=20191108, decision_function_shape='ovr', max_iter=-1)
subset1 = train_Data[train_Data['group'] == 0]
subset1 = subset1.append(train_Data[train_Data['group'] == 1])
train_subset1 = subset1[['radius', 'theta']]
svm_1.fit(train_subset1, subset1['SpectralCluster'])

# build SVM 1: Group 1 versus Group 2
svm_2 = svm.SVC(kernel="linear", random_state=20191108, decision_function_shape='ovr', max_iter=-1)
subset2 = train_Data[train_Data['group'] == 1]
subset2 = subset2.append(train_Data[train_Data['group'] == 2])
train_subset2 = subset2[['radius', 'theta']]
svm_2.fit(train_subset2, subset2['SpectralCluster'])

# build SVM 2: Group 2 versus Group 3
svm_3 = svm.SVC(kernel="linear", random_state=20191108, decision_function_shape='ovr', max_iter=-1)
subset3 = train_Data[train_Data['group'] == 2]
subset3 = subset3.append(train_Data[train_Data['group'] == 3])
train_subset3 = subset3[['radius', 'theta']]
svm_3.fit(train_subset3, subset3['SpectralCluster'])

print("Equation of the separating hyperplane for SVM 0:")
print("𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎")
print(f'({numpy.round(svm_1.intercept_[0] ,7)})'
    f' + ({numpy.round(svm_1.coef_[0][0], 7)}*x_1)'
    f' + ({numpy.round(svm_1.coef_[0][1], 7)}*x_2) = 𝟎')
print("")

print("Equation of the separating hyperplane for SVM 1:")
print("𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎")
print(f'({numpy.round(svm_2.intercept_[0] ,7)})'
    f' + ({numpy.round(svm_2.coef_[0][0], 7)}*x_1)'
    f' + ({numpy.round(svm_2.coef_[0][1], 7)}*x_2) = 𝟎')
print("")

print("Equation of the separating hyperplane for SVM 2:")
print("𝑤_0+𝐰_1*𝐱_1+w_2*x_2=𝟎")
print(f'({numpy.round(svm_3.intercept_[0] ,7)})'
    f' + ({numpy.round(svm_3.coef_[0][0], 7)}*x_1)'
    f' + ({numpy.round(svm_3.coef_[0][1], 7)}*x_2) = 𝟎')


# g) (5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code
# the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the
# hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart
# title.  Also, grid lines should be added to the axes.

# In[39]:


# getting hyperplanes for all the SVM
w = svm_1.coef_[0]
a = -w[0] / w[1]
xx1 = numpy.linspace(1, 4)
yy1 = a * xx1 - (svm_1.intercept_[0]) / w[1]
w = svm_2.coef_[0]
a = -w[0] / w[1]
xx2 = numpy.linspace(1, 4)
yy2 = a * xx2 - (svm_2.intercept_[0]) / w[1]
w = svm_3.coef_[0]
a = -w[0] / w[1]
xx3 = numpy.linspace(1, 4)
yy3 = a * xx3 - (svm_3.intercept_[0]) / w[1]
# plot polar coordinates and hyperplanes
for i in range(4):
    x_y = train_Data[train_Data['group'] == i]
    plt.scatter(x_y['radius'], x_y['theta'], c=color_array[i], label=i)
plt.plot(xx1, yy1, color='black', linestyle='-')
plt.plot(xx2, yy2, color='black', linestyle='-')
plt.plot(xx3, yy3, color='black', linestyle='-')
plt.xlabel('Radius ------->')
plt.ylabel('Theta ------>')
plt.legend(title='Group', loc='best', )
plt.grid(True)
plt.show()
print('Support Vector Machines on Four Segments')


# h) (10 points) Convert the observations along with the hyperplanes from the polar coordinates back to the
# Cartesian coordinates. Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code
# the points using the SpectralCluster (0 = Red and 1 = Blue). Besides, plot the hyper-curves as dotted lines to the
# graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also,
# grid lines should be added to the axes. Based on your graph, which hypercurve do you think is not needed?

# In[46]:


#Polar coordinates to the Cartesian coordinates
h1_xx1 = xx1 * numpy.cos(yy1)
h1_yy1 = xx1 * numpy.sin(yy1)
h2_xx2 = xx2 * numpy.cos(yy2)
h2_yy2 = xx2 * numpy.sin(yy2)
h3_xx3 = xx3 * numpy.cos(yy3)
h3_yy3 = xx3 * numpy.sin(yy3)

color_array = ['red', 'blue']
for i in range(2):
    x_y = train_Data[train_Data['SpectralCluster'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.plot(h1_xx1, h1_yy1, color='green', linestyle='-')
plt.plot(h2_xx2, h2_yy2, color='black', linestyle='-')
plt.plot(h3_xx3, h3_yy3, color='black', linestyle='-')
plt.xlabel('x ------>')
plt.ylabel('y ------>')
plt.legend(title='Spectral_Cluster', loc='best')
plt.grid(True)
plt.show()
print('       Support Vector Machines on Two Segments')
print("")

