# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:37:28 2018

@author: suhas
"""

import numpy as np
import pandas as pd

import sklearn.naive_bayes as NB

dataframe =  pd.read_csv("Purchase_Likelihood.csv")

X = pd.DataFrame(dataframe, columns=['group_size','homeowner', 'married_couple'])
y = pd.DataFrame(dataframe, columns=['A'])

classifier = NB.MultinomialNB().fit(X, y)

class_counts = classifier.class_count_
total = sum(class_counts)
print("Class 1 probability",class_counts[0]/total)
print("Class 2 probability",class_counts[1]/total)
print("Class 3 probability",class_counts[2]/total)

q2 = np.array([[1,0,0], [2,1,1], [3,1,1], [4,0,0]])

predProb = classifier.predict_proba(q2)
print('Predicted Probability :\n', predProb)


all = np.array([[1,0,0], 
              [1,1,1], 
              [1,1,0], 
              [1,0,1],
              [2,0,0], 
              [2,1,1], 
              [2,1,0], 
              [2,0,1],
              [3,0,0], 
              [3,1,1], 
              [3,1,0], 
              [3,0,1],
              [4,0,0], 
              [4,1,1], 
              [4,1,0], 
              [4,0,1]])

predProbAll = classifier.predict_proba(all)
print('Predicted Probability All :\n', predProbAll)