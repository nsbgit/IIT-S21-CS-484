# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:16:46 2021

@author: Sukanta
"""

# Importig Libraries
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.metrics as metrics
import graphviz

# -------------------------------

# global variables
SPLITTING_CRITERION = 'entropy'
MAXIMUM_TREE_DEPTH = 5
INIT_RNDM_SEED = 20210415
MAX_BOOSTING_ITR = 50
INTERRUPT_ACCURACY = 0.9999999
INPUT_FEATURES = ['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']
TARGET_FEATURE = 'quality_grp'
# -------------------------------

# Read Data -------------------------------
trainData = pandas.read_csv('WineQuality_Train.csv')
testData = pandas.read_csv('WineQuality_Test.csv')
# -------------------------------
nObs = trainData.shape[0]

x_train = trainData[INPUT_FEATURES]
y_train = trainData[TARGET_FEATURE]

x_test = trainData[INPUT_FEATURES]
y_test = trainData[TARGET_FEATURE]
# q1.a, b -------------------------------
# classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
# treeFit = classTree.fit(x_train, y_train)
# treePredProb = classTree.predict_proba(x_train)
# accuracy = classTree.score(x_train, y_train)
# print(f'Misclassification Rate Iteration 0 = {1-accuracy}')

# # q1.b -------------------------------


w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(50)
ensemblePredProb = numpy.zeros((nObs, 2))
is_converged = False
coverged_accuracy = numpy.nan
covereged_iteration = numpy.nan

for iter in range(50):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy[iter] = classTree.score(x_train, y_train, w_train)
    ensemblePredProb += accuracy[iter] * treePredProb

    if (accuracy[iter] >= INTERRUPT_ACCURACY):
        is_converged = True
        coverged_accuracy = accuracy[iter]
        covereged_iteration = iter
        break
    
    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - treePredProb[:,1]), (treePredProb[:,1]))
    predClass = numpy.where(treePredProb[:,1] >= 0.2, 1, 0)
    w_train = numpy.where(predClass != y_train, 2+numpy.abs(eventError), numpy.abs(eventError))

misclass=1-accuracy

print(f'The Misclassification Rate of the classification tree on the Training data at Iteration  0 = {misclass[0]}')
print(f'The Misclassification Rate of the classification tree on the Training data at Iteration  1 = {misclass[1]}')


#  q1.c -------------------------------

# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
