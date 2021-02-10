# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:01:37 2020

@author: Lipi
"""

import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\FinalExam\WineQuality_Train.csv')
test_data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\FinalExam\WineQuality_Test.csv')

x_train = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = train_data['quality_grp']

x_test = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = test_data['quality_grp']

w_train = np.array([1 for i in range(len(x_train))])


for iter in range(50):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=None)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy = classTree.score(x_train, y_train, w_train)
    print('Accuracy = ', accuracy)

    # Update the weights
    eventError = np.empty((len(x_train), 1))
    predClass = np.empty((len(x_train), 1))

    for i in range(len(x_train)):
        if (y_train[i] == 0):
            eventError[i] = treePredProb[i,1]
        else:
            eventError[i] = 1 - treePredProb[i,1]

        if (treePredProb[i,1] >= treePredProb[i,0]):
            predClass[i] = 1
        else:
            predClass[i] = 0

        if (predClass[i] != y_train[i]):
            w_train[i] = 2 + np.abs(eventError[i])
        else:
            w_train[i] = np.abs(eventError[i])

    print('Event Error:\n', eventError)
#     print('Predicted Class:\n', predClass)
#     print('Weight:\n', w_train)   

    if accuracy >= 0.9999999:
        break
    
    


treeFit = classTree.fit(x_test, y_test)
accuracy = classTree.score(x_test, y_test)
accuracy