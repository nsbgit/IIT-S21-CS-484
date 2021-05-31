#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[1]:


#prob1
import pandas as pd
data = pd.read_csv("SpiralWithCluster.csv") 
spir=data['SpectralCluster']
print(data)
spir.value_counts()


# In[2]:


#xVar = data[coloums=['x','y']
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.neural_network as nn
import sklearn.metrics as metrics
xVar=data[['x', 'y']] 
y=data['SpectralCluster']


# In[3]:


def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)

    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    Accuracy=1-metrics.accuracy_score(y, y_pred)
    
    # Plot the prediction
#     plt.figure(figsize=(10,6))
#     plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
#     plt.plot(xVar, y_pred, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
#     plt.grid(True)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
#     plt.legend(fontsize = 12, markerscale = 3)
#     plt.show()
    
    return (Loss, RSquare, Accuracy)

result = pd.DataFrame(columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
act=["identity", "logistic", "tanh", "relu"]

for i in numpy.arange(1,6):
    for j in numpy.arange(1,11):
        Loss, RSquare, Accuracy = Build_NN_Toy (nLayer = i, nHiddenNeuron = j)
        result = result.append(pandas.DataFrame([['relu',i, j, Loss, RSquare,Accuracy]], 
                               columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']))
result.to_csv(r'relu.csv')


# In[4]:


def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'identity', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)

    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    Accuracy=1-metrics.accuracy_score(y, y_pred)
    
    # Plot the prediction
#     plt.figure(figsize=(10,6))
#     plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
#     plt.plot(xVar, y_pred, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
#     plt.grid(True)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
#     plt.legend(fontsize = 12, markerscale = 3)
#     plt.show()
    
    return (Loss, RSquare, Accuracy)

result = pd.DataFrame(columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
act=["identity", "logistic", "tanh", "relu"]

for i in numpy.arange(1,6):
    for j in numpy.arange(1,11):
        Loss, RSquare, Accuracy = Build_NN_Toy (nLayer = i, nHiddenNeuron = j)
        result = result.append(pandas.DataFrame([['identity',i, j, Loss, RSquare,Accuracy]], 
                               columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']))
result.to_csv(r'identity.csv')


# In[5]:


import matplotlib
def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)

    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    Accuracy=1-metrics.accuracy_score(y, y_pred)
    
    # Plot the prediction
    #plt.figure(figsize=(10,6))
    #plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
    matplotlib.pyplot.scatter(xVar['x'], y_pred)
    return (Loss, RSquare, Accuracy)

result = pd.DataFrame(columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
act=["identity", "logistic", "tanh", "relu"]
Loss, RSquare, Accuracy = Build_NN_Toy (nLayer = 1, nHiddenNeuron = 2)


# In[6]:


#logistic
def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'logistic', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)

    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    Accuracy=1-metrics.accuracy_score(y, y_pred)
    
    # Plot the prediction
#     plt.figure(figsize=(10,6))
#     plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
#     plt.plot(xVar, y_pred, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
#     plt.grid(True)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
#     plt.legend(fontsize = 12, markerscale = 3)
#     plt.show()
    
    return (Loss, RSquare, Accuracy)

result = pd.DataFrame(columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
act=["identity", "logistic", "tanh", "relu"]

for i in numpy.arange(1,6):
    for j in numpy.arange(1,11):
        Loss, RSquare, Accuracy = Build_NN_Toy (nLayer = i, nHiddenNeuron = j)
        result = result.append(pandas.DataFrame([['logistic',i, j, Loss, RSquare,Accuracy]], 
                               columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']))
result.to_csv(r'logistic.csv')


# In[7]:


#tanh
def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'tanh', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)

    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    Accuracy=1-metrics.accuracy_score(y, y_pred)
    
    # Plot the prediction
#     plt.figure(figsize=(10,6))
#     plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
#     plt.plot(xVar, y_pred, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
#     plt.grid(True)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
#     plt.legend(fontsize = 12, markerscale = 3)
#     plt.show()
    
    return (Loss, RSquare, Accuracy)

result = pd.DataFrame(columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
act=["identity", "logistic", "tanh", "relu"]

for i in numpy.arange(1,6):
    for j in numpy.arange(1,11):
        Loss, RSquare, Accuracy = Build_NN_Toy (nLayer = i, nHiddenNeuron = j)
        result = result.append(pandas.DataFrame([['tanh',i, j, Loss, RSquare,Accuracy]], 
                               columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']))
result.to_csv(r'tanh.csv')


# In[8]:


#combining all csv
c1 = pd.read_csv("identity.csv")
c1=c1[['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']]

c2 = pd.read_csv("tanh.csv")
c2=c2[['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']]

c3 = pd.read_csv("relu.csv")
c3=c3[['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']]

c4 = pd.read_csv("logistic.csv")
c4=c4[['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']]

# c1.loc[c1.Misclassification >=0.5, 'trying'] = 'Hi'
# c1.head()
c1.head()

c11 = c1[c1.Misclassification <= 0.5]
# c11.head()


c22 = c2[c2.Misclassification <= 0.5]
# c22.head()

c33 = c3[c3.Misclassification <= 0.5]
# c33.head()

c44 = c4[c4.Misclassification <= 0.5]
c44.head()
x=[c11,c22,c33,c44]
out=[]
result = pd.DataFrame(columns = ['Number of iteration','Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
for i in x:
    o1=(i.loc[i['Loss'].idxmin()]).name
    o2=(i.loc[i['Loss'].idxmin()]).values.tolist()
    print(o2)
    result = result.append(pandas.DataFrame([[o1+1,o2[0],o2[1],o2[2],o2[3],o2[4],o2[5]]], 
                               columns = ['Number of iteration','Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']))

result=result[['Number of iteration','Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification']]
print(result)

#c44['Loss'].min()
#c44.loc[c44['Loss'].idxmin()]


# In[9]:


c3.head()


# In[10]:


#trying scatterplot
import statistics as st
def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                            activation = 'relu', verbose = False,
                            solver = 'lbfgs', learning_rate_init = 0.1,
                            max_iter = 5000, random_state = 20191108)
    # nnObj.out_activation_ = 'identity'
    thisFit = nnObj.fit(xVar, y) 
    y_pred = nnObj.predict(xVar)
    yproba=nnObj.predict_proba(xVar)
    #print(yproba)
    #print(st.stdev(y_pred))
    Loss = nnObj.loss_
    RSquare = metrics.r2_score(y, y_pred)
    Accuracy=1-metrics.accuracy_score(y, y_pred)
    
    
    # Plot the prediction
#     plt.figure(figsize=(10,6))
#     plt.plot(xVar, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
#     plt.plot(xVar, y_pred, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
#     plt.grid(True)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
#     plt.legend(fontsize = 12, markerscale = 3)
#     plt.show()
    
    return (Loss, RSquare, Accuracy,y_pred,yproba)

result = pd.DataFrame(columns = ['Activation','nLayer', 'nHiddenNeuron', 'Loss', 'RSquare','Misclassification'])
act=["identity", "logistic", "tanh", "relu"]
Loss, RSquare, Accuracy,col,pred_prob = Build_NN_Toy (nLayer = 4, nHiddenNeuron = 8)
color=[]
spec1=[]
spec0=[]
pred_prob=list(pred_prob)
for i in range(len(col)):
    if(col[i]==0):
        color.append('Red')
        spec0.append(pred_prob[i])
    else:
        color.append('Blue')
        spec1.append(pred_prob[i])
plt.scatter(data['x'],data['y'], c=color, alpha=0.5)
plt.show()
#print(spec1)
xspec=[]
yspec=[]
for i in spec1:
    xspec.append(i[0])
    yspec.append(i[1])
#print(xspec)
tt=st.mean(xspec)
print(format(tt, '.10f'))
print(format(st.stdev(yspec), '.10f'))
#st.stdev(yspec)


# In[11]:


print(data['x'])


# In[12]:


#Question2
from sklearn import svm
xTrain=data[['x', 'y']] 
yTrain=data['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20191106, max_iter=-1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)
print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
acc=metrics.accuracy_score(yTrain,y_predictClass)
print("Misclassficiation rate",1-acc)
data['_PredictedClass_'] = y_predictClass
svm_Mean = data.groupby('_PredictedClass_').mean()

# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
print("Coefficients ",w)
# xx = numpy.linspace(-3, 3)
# yy = a * xx - (thisFit.intercept_[0]) / w[1]

# # plot the parallels to the separating hyperplane that pass through the
# # support vectors
# b = thisFit.support_vectors_[0]
# yy_down = a * xx + (b[1] - a * b[0])

# b = thisFit.support_vectors_[-1]
# yy_up = a * xx + (b[1] - a * b[0])
print("The equation is ",thisFit.intercept_[0],"+",w[0],"x +", w[1],"y = 0")


# In[54]:


import numpy as np
a=w[0]
b=w[1]
x = np.linspace(-5,5,100)
c=thisFit.intercept_[0]
y=-(a*x-c)/b
plt.plot(x, y, ':')
plt.scatter(data['x'],data['y'], c=color, alpha=0.5)


# In[14]:


#trying Q2
xTrain=data[['x', 'y']] 
yTrain=data['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear',decision_function_shape = 'ovr',random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))

data['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)
#print(thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
for j in range(len(thisFit.coef_)):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]
    #print(yy)


# In[15]:


a=thisFit.coef_[0][0]
b=thisFit.coef_[0][1]
x = numpy.linspace(-6, 6)
y = numpy.zeros((len(x),3))
c=thisFit.coef_
#y=(c-(a*x))/b
for j in range(len(thisFit.coef_)):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    y[:,j] = a * x - (thisFit.intercept_[j]) / w[1]
    #print(yy)
plt.plot(x, y, ':')
plt.scatter(data['x'],data['y'], c=color, alpha=0.5)


# In[16]:


#2d
# Convert to the polar coordinates
data['radius'] = numpy.sqrt(data['x']**2 + data['y']**2)
data['theta'] = numpy.arctan2(data['y'],data['x'])

# ArcTan2 gives angle from –Pi to Pi
# Make the angle from 0 to 2*Pi
def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

data['theta'] = data['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = data[['radius','theta']]
yTrain = data['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
data['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)
print(len(thisFit.coef_))
a=thisFit.coef_[0][0]
b=thisFit.coef_[0][1]
x = numpy.linspace(-3, 3)
y = numpy.zeros((len(x),3))
c=thisFit.coef_
#y=(c-(a*x))/b
for j in range(len(thisFit.coef_)):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    y[:,j] = a * x - (thisFit.intercept_[j]) / w[1]
    #print(yy)

#print(y)
# Back-transform the hyperplane from the Polar coordinates to the Cartesian coordinates

# h0_xx = x * numpy.cos(y[:,0])
# h0_yy = x * numpy.sin(y[:,0])

# h1_xx = x * numpy.cos(y[:,1])
# h1_yy = x * numpy.sin(y[:,1])


#
plt.axis([-2,5,0,10])
#ax.set(xlim=(0, 5), ylim=(0, 5))
plt.plot(x,y, ':')
plt.scatter(data['radius'],data['theta'], c=color, alpha=0.5)
#plt.scatter(h0_xx,h0_yy, c=color, alpha=0.5)
# print(data['radius'])
# print(data['theta'])


# In[ ]:





# In[17]:


from sklearn.cluster import KMeans
from collections import Counter
X=[]
for i in range(len(data['theta'])):
    x=[]
    x.append(data['radius'][i])
    x.append(data['theta'][i])
    X.append(x)
# print(X)
#kmeans = KMeans(n_clusters=4, random_state=20191108).fit(X)
# Counter(kmeans.labels_)
# print(kmeans.labels_)
# y_kmeans = kmeans.predict(X)
# print(y_kmeans)
#plt.plot(x,y, ':')
plt.scatter(data['radius'],data['theta'], c=color, alpha=0.5)

#centers = kmeans.cluster_centers_
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[18]:


#print(X)
from collections import Counter
newgrp=[]
for i in range(100):
    #print(X[i][0])
    if(X[i][0]<1.7 and X[i][1]>6 and col[i]==0):
        newgrp.append(0)
    elif(X[i][0]>0 and X[i][0]<3 and X[i][1]>2.8 and col[i]==1):
        newgrp.append(1)
    elif(X[i][0]>2.5 and X[i][1]<3.2 and col[i]==1):
        newgrp.append(3)
    else:
        newgrp.append(2)
print(newgrp)
Counter(newgrp)
data['Group']=newgrp


# In[19]:


newcolor=[]
for i in newgrp:
    if(i==0):
        newcolor.append('red')
    elif(i==1):
        newcolor.append('blue')
    elif(i==2):
        newcolor.append('green')
    else:
        newcolor.append('black')
plt.scatter(data['radius'],data['theta'], c=newcolor, alpha=0.5)


# In[20]:


data['Group']


# In[21]:


#data = data[data['_PredictedClass_'] == 1]
subData0 = data[data['Group'] == 1]
subData1 = data[data['Group'] == 0]
subData = pd.concat([subData0, subData1])
print(subData)
xTrain = subData[['x','y']]
yTrain = subData['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
subData['_PredictedClass_01'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
print(len(thisFit.coef_))
for j in range(len(thisFit.coef_)):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
# carray = ['red', 'green', 'blue']
#plt.figure(figsize=(10,10))
#for i in range(3):
# subData = data[data['_PredictedClass_'] == 0]
# plt.scatter(x = subData['x'],y = subData['y'],  label = (i+1), s = 25)
# plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
# plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
# plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
# plt.grid(True)
# plt.title('Required Graph')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(-7,7)
# plt.ylim(-7,7)
# plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
# plt.show()


# In[22]:


#data = data[data['_PredictedClass_'] == 1]
subData0 = data[data['Group'] == 2]
subData1 = data[data['Group'] == 1]
subData = pd.concat([subData0, subData1])
print(subData)
xTrain = subData[['x','y']]
yTrain = subData['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
subData['_PredictedClass_01'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
print(len(thisFit.coef_))
for j in range(len(thisFit.coef_)):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]

#print(yy)
# plot the line, the points, and the nearest vectors to the plane
# carray = ['red', 'green', 'blue']
#plt.figure(figsize=(10,10))
#for i in range(3):
# subData = data[data['_PredictedClass_'] == 0]
# plt.scatter(x = subData['x'],y = subData['y'],  label = (i+1), s = 25)
# plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
# plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
# plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
# plt.grid(True)
# plt.title('Required Graph')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(-7,7)
# plt.ylim(-7,7)
# plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
# plt.show()


# In[23]:


#data = data[data['_PredictedClass_'] == 1]
subData0 = data[data['Group'] == 3]
subData1 = data[data['Group'] == 2]
subData = pd.concat([subData0, subData1])
print(subData)
xTrain = subData[['x','y']]
yTrain = subData['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
subData['_PredictedClass_01'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
print(len(thisFit.coef_))
for j in range(len(thisFit.coef_)):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
# carray = ['red', 'green', 'blue']
#plt.figure(figsize=(10,10))
#for i in range(3):
# subData = data[data['_PredictedClass_'] == 0]
# plt.scatter(x = subData['x'],y = subData['y'],  label = (i+1), s = 25)
# plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
# plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
# plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
# plt.grid(True)
# plt.title('Required Graph')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(-7,7)
# plt.ylim(-7,7)
# plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
# plt.show()


# In[43]:


interc=[1.00003674,0.99999888,-1.92985156]
coeff=[[-0.00024189,-0.00293195],[0.0000049095,0.0000014375],[0.02273196,1.02904755]]
#print(format(coeff[1][1], '.10f'))
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
#print(len(thisFit.coef_))
for j in range(len(interc)):
    w = coeff[j]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (interc[j]) / w[1]
print(yy)
plt.axis([-2,100,0,100])
plt.scatter(x = data['radius'],y = data['theta'],s = 25)
plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')


# In[52]:


#trying 2g
ploty=[]
import numpy as np
for i in range(3):
    a=coeff[i][0]
    b=coeff[i][1]
    x = np.linspace(-5,5)
    c=interc[i]
    y=a*x-c/b
    ploty.append(y)
print(ploty)
#plt.axis([-100,100,-100000,100])
#plt.plot(x, ploty[0], ':',c="black")
#plt.plot(x, ploty[1], ':')
plt.plot(x, ploty[2], ':')
plt.scatter(data['radius'],data['theta'], c=newcolor, alpha=0.5)


# In[ ]:




