# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:50:22 2020

@author: Lipi
"""

## Note:- I have taken reference from the professor sample code for some part of assignment.
########################### Import Statements #################################
#################################################################################

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neural_network as nn
import sklearn.svm as svm
import statsmodels.api as sm


trainData = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week12_assignment5\\SpiralWithCluster.csv')

######################################### Question 1 A ########################################
scCounts = trainData['SpectralCluster'].value_counts()
percentObv = (scCounts[1] / (scCounts[0]+scCounts[1]))*100
print("The percentage of the observations have SpectralCluster = 1 is", percentObv)



###################################### Question 1 B ########################################

def mlpClassifier(actFunction):
    lossList = []
    result = pandas.DataFrame(columns = ['activation','niter','nLayer', 'nHiddenNeuron', 'Loss', 'misclassification'])
    for nLayer in numpy.arange(1,5):
        for nHiddenNeuron in numpy.arange(1,11,1):
            nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                                     activation = actFunction, verbose = False,
                                     solver = 'lbfgs', learning_rate_init = 0.1,
                                     max_iter = 5000, random_state = 20200408)
            
            thisFit = nnObj.fit(xTrain, yTrain) 
            y_predProb = nnObj.predict_proba(xTrain)
            y_pred = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
            activationfn = nnObj.activation
            mlp_loss = nnObj.loss_
            niter = nnObj.n_iter_
            activationOut = nnObj.out_activation_
            misclassification = 1 - (metrics.accuracy_score(yTrain, y_pred))
            result = result.append(pandas.DataFrame([[activationfn, niter, nLayer, nHiddenNeuron,mlp_loss, misclassification,activationOut]], 
                                    columns = ['activation','niter', 'nLayer', 'nHiddenNeuron', 'Loss', 'misclassification', 'output layer activation function']))
            
    return(result[result.Loss == result.Loss.min()])
    

xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']


# MLP Neural Network
y_threshold = trainData['SpectralCluster'].mean()
act1 = mlpClassifier('identity')
act2 = mlpClassifier('logistic')
act3 = mlpClassifier('tanh')
act4 = mlpClassifier('relu')

table = pandas.concat([act1,act2,act3,act4]).reset_index()
print(table)

#########################################Question 1 c ########################################

print("As per the answer in 1 b, Activation function of outlayer is Logistic")

#########################################Question 1 d ########################################

print(table[table.Loss == table.Loss.min()])
ans1d = table[table.Loss == table.Loss.min()]
ans1ddf = pandas.DataFrame(ans1d)


#########################################Question 1 e ########################################

xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']
 
nLayer = 4
nHiddenNeuron = 10
nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                         activation = 'relu', verbose = False,
                         solver = 'lbfgs', learning_rate_init = 0.1,
                         max_iter = 55, random_state = 20200408)
thisFit = nnObj.fit(xTrain, yTrain) 
y_predProb = nnObj.predict_proba(xTrain)
trainData['_PredictedClass_'] = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)


mlp_Mean = trainData.groupby('_PredictedClass_').mean()
print(mlp_Mean)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],                 y = subData['y'], c = carray[i], label = i, s = 25)
###############################need to check x y##################
plt.scatter(x = mlp_Mean['x'], y = mlp_Mean['y'], c = 'black', marker = 'X', s = 100)
###############################need to check x y##################
plt.grid(True)
plt.title('MLP (4 Layers, 10 Hidden Neurons)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#########################################Question 1 f ########################################

l = []
for i in range(len(y_predProb)):
    if trainData['SpectralCluster'][i] == 1:
        l.append(y_predProb[i][1])

print("count = ", len(l))
print("mean = ", numpy.mean(l))
print("std = ", numpy.std(l))


#########################################Question 2 A ########################################

svm_Model = svm.SVC(kernel = 'linear', random_state = 20200408, max_iter = -1, decision_function_shape  = 'ovr')
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)



print('Misclassification = ', 1 - (metrics.accuracy_score(yTrain, y_predictClass)))
trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)


print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)


#################################### Question 2 C #####################################
# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = numpy.linspace(-5, 5)
yy = a * xx - (thisFit.intercept_[0]) / w[1]


# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])



# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'], y = subData['y'], c = carray[i], label = i, s = 25)

plt.plot(xx, yy, color = 'black', linestyle = ':')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#################################### Question 2 D #####################################

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
                    random_state = 20200408, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)



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


#################################### Question 2 E #####################################
xTrain = trainData[['radius','theta']]
yTrain = trainData['SpectralCluster']
 #df.loc[df['shield'] > 6]
trainData['Group'] = 2

trainData.loc[(trainData['radius'] < 2.5) & (trainData['theta'] > 3) , ['Group']] = 1
trainData.loc[(trainData['radius'] < 3) & (trainData['theta'] > 5.5) & (trainData['radius'] > 2.5), ['Group']] = 1
trainData.loc[(trainData['radius'] > 2.5) & (trainData['theta'] < 2) , ['Group']] = 3
trainData.loc[(trainData['radius'] > 3.25) & (trainData['theta'] < 3.1) & (trainData['theta'] > 1.9), ['Group']] = 3
trainData.loc[(trainData['radius'] < 1.5) & (trainData['theta'] > 6), ['Group']] = 0
#trainData


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

############################### Question 2 F #####################################3
df01 = trainData[trainData['Group'].isin([0,1])]
xTrain = df01[['radius','theta']]
yTrain = df01['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20200408, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print("intercept = ", thisFit.intercept_)
print("coefficient = ", thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx01 = numpy.linspace(1, 5)
yy01 = a * xx01 - (thisFit.intercept_[0]) / w[1]


## SVM1


df12 = trainData[trainData['Group'].isin([1,2])]
xTrain = df12[['radius','theta']]
yTrain = df12['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20200408, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print("intercept = ", thisFit.intercept_)
print("coefficient = ", thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx12 = numpy.linspace(1, 5)
yy12 = a * xx12 - (thisFit.intercept_[0]) / w[1]


## SVM 2
df23 = trainData[trainData['Group'].isin([2,3])]
xTrain = df23[['radius','theta']]
yTrain = df23['Group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20200408, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print("intercept = ", thisFit.intercept_)
print("coefficient = ", thisFit.coef_)

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx23 = numpy.linspace(1, 5)
yy23 = a * xx23 - (thisFit.intercept_[0]) / w[1]


############################### Question 2 G #####################################3
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

############################ Question 2 H ###########################################3

h0_xx01 = xx01 * numpy.cos(yy01)
h0_yy01 = xx01 * numpy.sin(yy01)

h0_xx12 = xx12 * numpy.cos(yy12)
h0_yy12 = xx12 * numpy.sin(yy12)

h0_xx23 = xx23 * numpy.cos(yy23)
h0_yy23 = xx23 * numpy.sin(yy23)


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

