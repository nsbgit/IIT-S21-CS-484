#!/usr/bin/env python
# coding: utf-8

# In[43]:


import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neural_network as nn
import sklearn.svm as svm
import statsmodels.api as sm


# In[44]:


trainData = pandas.read_csv('D:\\IIT Edu\\Sem1\\MachineLearning\\Week12_assignment5\\SpiralWithCluster.csv')

y_threshold = trainData['SpectralCluster'].mean()
y_threshold


# In[45]:


#########################################Question 1 a ########################################
scCounts = trainData['SpectralCluster'].value_counts()
percentObv = (scCounts[1] / (scCounts[0]+scCounts[1]))*100
print("The percentage of the observations having SpectralCluster equals to 1 is", percentObv)


# In[46]:


xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']


# In[47]:


def mlpClassifier(actFunction):
    lossList = []
    result = pandas.DataFrame(columns = ['activation','niter','nLayer', 'nHiddenNeuron', 'Loss', 'misclassification'])
    for nLayer in numpy.arange(1,5):
        for nHiddenNeuron in numpy.arange(1,11,1):
            nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                                     activation = actFunction, verbose = False,
                                     solver = 'lbfgs', learning_rate_init = 0.1,
                                     max_iter = 5000, random_state = 20191108)
            
            thisFit = nnObj.fit(xTrain, yTrain) 
            y_predProb = nnObj.predict_proba(xTrain)
            y_pred = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
            activationfn = nnObj.activation
            mlp_loss = nnObj.loss_
            #lossList.append(mlp_loss)
            niter = nnObj.n_iter_
            activationOut = nnObj.out_activation_
            misclassification = 1 - (metrics.accuracy_score(yTrain, y_pred))
            result = result.append(pandas.DataFrame([[activationfn, niter, nLayer, nHiddenNeuron,mlp_loss, misclassification,activationOut]], 
                                   columns = ['activation','niter', 'nLayer', 'nHiddenNeuron', 'Loss', 'misclassification', 'output layer activation function']))
            #print(result) 
    return(result[result.Loss == result.Loss.min()])


# In[48]:



act1 = mlpClassifier('identity')
act2 = mlpClassifier('logistic')
act3 = mlpClassifier('tanh')
act4 = mlpClassifier('relu')


# In[49]:


table = pandas.concat([act1,act2,act3,act4])

table


# In[50]:


table[table.Loss == table.Loss.min()]


# In[51]:


xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']
 
nLayer = 4
nHiddenNeuron = 8
nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                         activation = 'relu', verbose = False,
                         solver = 'lbfgs', learning_rate_init = 0.1,
                         max_iter = 5000, random_state = 20191108)
thisFit = nnObj.fit(xTrain, yTrain) 
y_predProb = nnObj.predict_proba(xTrain)
trainData['_PredictedClass_'] = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)

mlp_Mean = trainData.groupby('_PredictedClass_').mean()
print(mlp_Mean)


# In[52]:


l = []
for i in range(len(y_predProb)):
    if trainData['SpectralCluster'][i] == 1:
        l.append(y_predProb[i][1])
        
print("count = ", len(l))
print("mean = ", numpy.mean(l))
print("std = ", numpy.std(l))


# In[55]:


carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'], y = subData['y'], c = carray[i], label = i, s = 25)
plt.scatter(x = mlp_Mean['x'], y = mlp_Mean['y'], c = 'black', marker = 'X', s = 100)
plt.grid(True)
plt.title('MLP (4 Layers, 8 Neurons)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


# In[ ]:





# In[ ]:




