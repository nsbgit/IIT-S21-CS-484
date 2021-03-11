# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:36:22 2021

@author: pc
"""

import pandas as pd
import statsmodels.api as stats
import numpy as np
from sklearn.metrics import accuracy_score

trainData = pd.read_csv('q15trainData.csv')

xTrain = trainData['x']
xTrain = stats.add_constant(xTrain, prepend=True)
yTrain = trainData['y']
model = stats.Logit(yTrain, xTrain)
thisFit = model.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

testData = pd.read_csv('q15testData.csv')
xTest = testData['x']
xTest = stats.add_constant(xTest, prepend=True)
yTest = testData['y']

thresold = 0.3

yPred = np.multiply(thisFit.predict(xTest) >= thresold, 1)
print("Misclassification Rate: {:.4f}".format(1 - accuracy_score(yTest, yPred)))