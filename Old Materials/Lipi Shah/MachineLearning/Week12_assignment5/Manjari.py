# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:51:10 2020

@author: Lipi
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neural_network as nn

SpiralData = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week12_assignment5\\SpiralWithCluster.csv')


#########################################Question 1 a ########################################

binarycount = SpiralData['SpectralCluster'].value_counts()

binarypercent = (binarycount[1] / (binarycount[0] + binarycount[1]))*100
# print("The percentage of observations having SpectralCluster equals to 1 : ", binarypercent)

#########################################Question 1 b ########################################

# MLP Neural Network

xTrain = SpiralData[['x','y']]
yTrain = SpiralData['SpectralCluster']

y_threshold = SpiralData['SpectralCluster'].mean()


def mlpClassifier(actFunction):

    result = pandas.DataFrame(columns = ['activation','niter','nLayer', 'nHiddenNeuron', 'Loss'])
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



            result = result.append(pandas.DataFrame([[activationfn, niter, nLayer, nHiddenNeuron,mlp_loss]],
                                   columns = ['activation','niter', 'nLayer', 'nHiddenNeuron', 'Loss']))

    return(result[result.Loss == result.Loss.min()])


act1 = mlpClassifier('identity')
act2 = mlpClassifier('logistic')
act3 = mlpClassifier('tanh')
act4 = mlpClassifier('relu')

table = pandas.concat([act1, act2, act3, act4])
print(table)




xTrain = SpiralData[['x', 'y']]
yTrain = SpiralData['SpectralCluster']

nLayer = 4
nHiddenNeuron = 10
nnObj = nn.MLPClassifier(hidden_layer_sizes=(nHiddenNeuron,) * nLayer,
                         activation='relu', verbose=False,
                         solver='lbfgs', learning_rate_init=0.1,
                         max_iter=5000, random_state=20200408)
thisFit = nnObj.fit(xTrain, yTrain)
y_predProb = nnObj.predict_proba(xTrain)
SpiralData['PredictedClass'] = numpy.where(y_predProb[:, 1] >= y_threshold, 1, 0)

mlp_Mean = SpiralData.groupby('PredictedClass').mean()
print(mlp_Mean)




l = []
for i in range(len(y_predProb)):
    if SpiralData['SpectralCluster'][i] == 1:
        l.append(y_predProb[i][1])

print("count = ", len(l))
print("mean = ", numpy.mean(l))
print("std = ", numpy.std(l))



#
carray = ['red', 'blue']
plt.figure(figsize=(10, 10))
for i in range(2):
    subData = SpiralData[SpiralData['PredictedClass'] == i]
    plt.scatter(x=subData['x'], y=subData['y'], c=carray[i], label=i, s=25)
plt.scatter(x=mlp_Mean['x'], y=mlp_Mean['y'], c='black', marker='X', s=100)
plt.grid(True)
plt.title('MLP (4 Layers, 10 Neurons)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title='Predicted Class', loc='best', bbox_to_anchor=(1, 1), fontsize=14)
plt.show()
#
#
#
#
#