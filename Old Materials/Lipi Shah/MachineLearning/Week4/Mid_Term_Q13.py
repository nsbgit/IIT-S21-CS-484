# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:51:17 2020

@author: Lipi
"""



import numpy as np
import pandas as pd
import scipy
import statsmodels.api as stats

train_data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4\\Que_13_train.csv')
cols1 = ['X']

test_data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4\\Que3_Test.csv')

X = np.where(train_data['Y'].notnull(), 1, 0)
DF0 = np.linalg.matrix_rank(X)
y = train_data['Y']

# Intercept
logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)


# Intercept + CREDIT_SCORE_BAND
Credit_Score = train_data[['X']].astype('category')
X = pd.get_dummies(Credit_Score)
X = stats.add_constant(X, prepend=True)
DF1 = np.linalg.matrix_rank(X) 

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

Deviance = 2 * (LLK1 - LLK0)
DF = DF1 - DF0
pValue = scipy.stats.chi2.sf(Deviance, DF)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)
print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)


# Train data
X_train = train_data[['X']]
X_train = stats.add_constant(X_train, prepend=True)

Y_train = train_data['Y']

# Test Data
X_test = test_data[['X']]
X_test = stats.add_constant(X_test, prepend=True)

Y_test = test_data['Y']

###################3
threshold = train_data['Y'].value_counts()[1] / len(train_data)
threshold

######################33
logit = stats.MNLogit(Y_train, X_train)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

predicted_probabilities = thisFit.predict(X_test)
predicted_probabilities


# In[ ]:


nY = 5
# Calculate the Root Average Squared Error
RASE = 0.0
for i in range(nY):
    if (Y_test[i] == 1):
        RASE += (1 - predicted_probabilities[1][i])**2
    else:
        RASE += (0 - predicted_probabilities[1][i])**2
RASE = np.sqrt(RASE/nY)

print("Root Average Square Error = ",RASE)
       


# In[ ]:

