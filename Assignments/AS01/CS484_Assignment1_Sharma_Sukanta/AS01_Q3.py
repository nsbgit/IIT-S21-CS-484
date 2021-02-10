# -*- coding: utf-8 -*-
"""

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Spring 2021
"""

import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
import scipy
from numpy import linalg as LA
from scipy import linalg as LA2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors as kNN
 
     
# **************************  Question 3  ******************************************
fraud = pd.read_csv('Fraud.csv',delimiter=',')

# **************************  Question 3.a  ******************************************
fraudCount = fraud[fraud['FRAUD'] == 1].count()['FRAUD']
totalCount = fraud['FRAUD'].count()
fraudPercentage = (fraudCount / totalCount) * 100
fraudPercentage = float("{0:.4f}".format(fraudPercentage))
print("\n\n\nQ3.a)	(5 points) What percent of investigations are found to be frauds?  Please give your answer up to 4 decimal places.\n")
print("\n{0}%\n".format(fraudPercentage))

# **************************  Question 3.b.i  ******************************************
print("\n\n\nQ3.b.i)   (5 points) How many dimensions are used?\n")
fraudList = fraud.iloc[:,[2,3,4,5,6,7]] 
fraudList= np.matrix(fraudList)
xtx = fraudList.transpose() * fraudList
#print(xtx)
evals, evecs = LA.eigh(xtx)
# Want eigenvalues greater than one
evals_1 = evals[evals > 1.0]
evecs_1 = evecs[:,evals > 1.0]
print("Eigenvalues of x = \n", evals_1)
print(evals > 1)




# **************************  Question 3.b.ii  ******************************************
print("\n\n\nQ3.b.ii)   (5 points) Please provide the transformation matrix?  Show evidence that the orthonormalized columns are actually orthonormal.\n")

transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)
transf_fraudList = fraudList * transf;
print("The Transformed x = \n", transf_fraudList)
xtx = transf_fraudList.transpose() * transf_fraudList;
print("Expecting an Identity Matrix, so the resulting variables are orthonormal= \n", xtx)





# **************************  Question 3.c.ii  ******************************************
print("\n\n\nQ3.c.ii)   (5 points) Run the score function, show and explain the function return value.\n")
trainData = fraudList
target = fraud['FRAUD']
neigh = KNeighborsClassifier(n_neighbors=5 , metric = 'euclidean', algorithm = 'brute')
nbrs = neigh.fit(trainData, target)
scoreValue = nbrs.score(trainData, target) 
print("\n\nScore function values:{0}\n".format(scoreValue))
missclassification = (1 -scoreValue) * 100
print("Misclassification = {0}".format(missclassification))






# **************************  Question 3.d  ******************************************
print("\n\n\nQ3.d)   (5 points) For the observation which has these input variable values: TOTAL_SPEND = 7500, DOCTOR_VISITS = 15, NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2, find its five neighbors.  Please list their input variable values and the target values. Reminder: transform the input observation using the results in (b) before finding the neighbors")
focal_point = [7500, 15, 3, 127, 2, 2]
kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
#transformation matrix
transf_focal_input = focal_point * transf;
print("\nTrasformed input \n", transf_focal_input)
nbrs_trans = kNNSpec.fit(transf_fraudList) # fitting
distance, index = nbrs_trans.kneighbors(transf_fraudList)


targetNeighbr = nbrs_trans.kneighbors(transf_focal_input, return_distance = False)

print("Neighbhr of target input values = \n", targetNeighbr)
nhbrd = fraud.iloc[list(targetNeighbr[0])]
print("Nearest Neighbhr of taget case id and value = \n", nhbrd)



# **************************  Question 3.e  ******************************************
print("\n\n\nQ3.e)   (5 points) Follow-up with (d), what is the predicted probability of fraud (i.e., FRAUD = 1)?  If your predicted probability is greater than or equal to your answer in (a), then the observation will be classified as a fraud.  Otherwise, not a fraud.  Based on this criterion, will the observation in (d) be misclassified?")
fraud_pred = nbrs.predict(transf_fraudList)
print("Fraud Prediction Matrix", fraud_pred)
    
    
    
    
    
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
# **************************  Question 3  ******************************************
    
    
   
    
    
    
    
    
    
    
   
   
   
    
    
     
    
    