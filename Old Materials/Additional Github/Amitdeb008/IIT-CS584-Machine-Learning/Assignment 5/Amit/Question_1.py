#!/usr/bin/env python
# coding: utf-8

# Assignment 5

# In[20]:


# Import Libraries
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neural_network as nn
import sklearn.svm as svm
import statsmodels.api as sm


# In[23]:


train_Data = pandas.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 5\\SpiralWithCluster.csv')

y_thrsld = train_Data['SpectralCluster'].mean()


# Question 1

# a) (5 points) What percent of the observations have SpectralCluster equals to 1?

# In[26]:


sc_Counts = train_Data['SpectralCluster'].value_counts()
percent_Ob= (sc_Counts[1] / (sc_Counts[0]+sc_Counts[1]))*100
print(percent_Ob,"% percent of the observations have SpectralCluster equals to 1")


# b) (15 points) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.  You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1. Your search will be done over a grid that is formed by cross-combining the following attributes: (1) activation function: identity, logistic, relu, and tanh; (2) number of hidden layers: 1, 2, 3, 4, and 5; and (3) number of neurons: 1 to 10 by 1.  List your optimal neural network for each activation function in a table.  Your table will have four rows, one for each activation function.  Your table will have five columns: (1) activation function, (2) number of layers, (3) number of neurons per layer, (4) number of iterations performed, (5) the loss value, and (6) the misclassification rate.

# In[28]:


x_Train = train_Data[['x','y']]
y_Train = train_Data['SpectralCluster']

def mlpClassifier(actFunction):
    lossList = []
    result = pandas.DataFrame(columns = ['activation','niter','nLayer', 'nHiddenNeuron', 'Loss', 'misclassification'])
    for nLayer in numpy.arange(1,5):
        for nHiddenNeuron in numpy.arange(1,11,1):
            nn_Obj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                                     activation = actFunction, verbose = False,
                                     solver = 'lbfgs', learning_rate_init = 0.1,
                                     max_iter = 5000, random_state = 20191108)
            thisFit = nn_Obj.fit(x_Train, y_Train) 
            y_predProb = nn_Obj.predict_proba(x_Train)
            y_pred = numpy.where(y_predProb[:,1] >= y_thrsld, 1, 0)
            activation_fn = nn_Obj.activation
            mlp_loss = nn_Obj.loss_
            niter = nn_Obj.n_iter_
            activationOut = nn_Obj.out_activation_
            misclassification = 1 - (metrics.accuracy_score(y_Train, y_pred))
            result = result.append(pandas.DataFrame([[activation_fn, niter, nLayer, nHiddenNeuron,mlp_loss, misclassification,activationOut]], 
                                   columns = ['activation','niter', 'nLayer', 'nHiddenNeuron', 'Loss', 'misclassification', 'output layer activation function']))
            
    return(result[result.Loss == result.Loss.min()])


act1 = mlpClassifier('identity')
act2 = mlpClassifier('logistic')
act3 = mlpClassifier('tanh')
act4 = mlpClassifier('relu')

table = pandas.concat([act1,act2,act3,act4]).reset_index()
print(table)


# c) (5 points) What is the activation function for the output layer?

# In[29]:


print("Activation function of outlayer is Logistic")


# d) (5 points) Which activation function, number of layers, and number of neurons per layer give the lowest loss and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are performed?

# In[30]:


print(table[table.Loss == table.Loss.min()])


# e) (5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[31]:


x_Train = train_Data[['x','y']]
y_Train = train_Data['SpectralCluster']
 
nLayer = 4
nHiddenNeuron = 8
nn_Obj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                         activation = 'relu', verbose = False,
                         solver = 'lbfgs', learning_rate_init = 0.1,
                         max_iter = 5000, random_state = 20191108)
thisFit = nn_Obj.fit(x_Train, y_Train) 
y_predProb = nn_Obj.predict_proba(x_Train)
train_Data['_PredictedClass_'] = numpy.where(y_predProb[:,1] >= y_thrsld, 1, 0)


mlp_Mean = train_Data.groupby('_PredictedClass_').mean()
print(mlp_Mean)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = train_Data[train_Data['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],                 y = subData['y'], c = carray[i], label = i, s = 25)
plt.scatter(x = mlp_Mean['x'], y = mlp_Mean['y'], c = 'black', marker = 'X', s = 100)
plt.grid(True)
plt.title('MLP (3 Layers, 7 Neurons)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()


# f)	(5 points) What is the count, the mean and the standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to the 10 decimal places.

# In[32]:


countPP = sum(y_predProb[:,1])
meanPP = numpy.mean(y_predProb[:,1])
stdPP = numpy.std(y_predProb[:,1])
print("count:              ",countPP)
print("mean:               ",meanPP)
print("standard deviation: ",stdPP)

