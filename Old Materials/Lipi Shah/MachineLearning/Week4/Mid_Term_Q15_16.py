#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import statsmodels.api as stats
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


# In[ ]:

#
data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week4\\policy_2001.csv')


# In[ ]:


data = data[['CLAIM_FLAG', 'CREDIT_SCORE_BAND', 'BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']]


# In[ ]:


train_data, test_data = train_test_split(data, test_size = 0.25, random_state = 20191009, stratify = data['CLAIM_FLAG'])


# In[ ]:


cols1 = ['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']


# In[ ]:


X = np.where(train_data['CLAIM_FLAG'].notnull(), 1, 0)
DF0 = np.linalg.matrix_rank(X) 

y = train_data['CLAIM_FLAG']


# In[ ]:


# Intercept
logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)


# In[ ]:


# Intercept + CREDIT_SCORE_BAND
Credit_Score = train_data[['CREDIT_SCORE_BAND']].astype('category')
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


# In[ ]:


def model1(column):
    X = train_data[[column]]
    X = stats.add_constant(X, prepend=True)
    DF1 = np.linalg.matrix_rank(X) 

    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK1 = logit.loglike(thisParameter.values)

    Deviance = 2 * (LLK1 - LLK0)
    DF = DF1 - DF0
    pValue = scipy.stats.chi2.sf(Deviance, DF)
    
    print('Summary of Intercept +', column)
    print(thisFit.summary())
    print("Model Log-Likelihood Value =", LLK1)
    print("Number of Free Parameters =", DF1)
    print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)
    print('\n')


# In[ ]:


for k in cols1:
    model1(k)


# In[ ]:


# Origin = Intercept + MVR_PTS
X = train_data[['MVR_PTS']]
X = stats.add_constant(X, prepend=True)
DF0 = np.linalg.matrix_rank(X) 

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)


# In[ ]:


# Intercept + MVR_PTS + CREDIT_SCORE_BAND
credit_score = train_data[['CREDIT_SCORE_BAND']].astype('category')
X = pd.get_dummies(credit_score)
X = X.join(train_data[['MVR_PTS']])
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


# In[ ]:


def model2(column):
    X = train_data[[column]]
    X = X.join(train_data[['MVR_PTS']])
    X = stats.add_constant(X, prepend=True)
    DF1 = np.linalg.matrix_rank(X) 

    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK1 = logit.loglike(thisParameter.values)

    Deviance = 2 * (LLK1 - LLK0)
    DF = DF1 - DF0
    pValue = scipy.stats.chi2.sf(Deviance, DF)

    print('Summary of Intercept + MVR_PTS +',column)
    print(thisFit.summary())
    print("Model Log-Likelihood Value =", LLK1)
    print("Number of Free Parameters =", DF1)
    print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)
    print('\n')


# In[ ]:


cols2 = ['BLUEBOOK_1000', 'CUST_LOYALTY', 'TIF', 'TRAVTIME']
for k in cols2:
    model2(k)


# In[ ]:


# Origin = Intercept + MVR_PTS + TRAVTIME
X = train_data[['MVR_PTS', 'TRAVTIME']]
X = stats.add_constant(X, prepend=True)
DF0 = np.linalg.matrix_rank(X) 

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK0 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK0)
print("Number of Free Parameters =", DF0)


# In[ ]:


# Intercept + MVR_PTS + TRAVTIME + CREDIT_SCORE_BAND
credit_score = train_data[['CREDIT_SCORE_BAND']].astype('category')
X = pd.get_dummies(credit_score)
X = X.join(train_data[['MVR_PTS', 'TRAVTIME']])
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


# In[ ]:


def model3(column):
    X = train_data[[column]]
    X = X.join(train_data[['MVR_PTS', 'TRAVTIME']])
    X = stats.add_constant(X, prepend=True)
    DF1 = np.linalg.matrix_rank(X) 

    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK1 = logit.loglike(thisParameter.values)

    Deviance = 2 * (LLK1 - LLK0)
    DF = DF1 - DF0
    pValue = scipy.stats.chi2.sf(Deviance, DF)

    print('Summary of Intercept + MVR_PTS + TRAVTIME +',column)
    print(thisFit.summary())
    print("Model Log-Likelihood Value =", LLK1)
    print("Number of Free Parameters =", DF1)
    print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)
    print('\n')


# In[ ]:


cols3 = ['BLUEBOOK_1000', 'CUST_LOYALTY', 'TIF']
for k in cols3:
    model3(k)


# Define the model with predictors as **MVR_PTS** and **TRAVTIME**

# In[ ]:


X_train = train_data[['MVR_PTS', 'TRAVTIME']]
X_train = stats.add_constant(X_train, prepend=True)

Y_train = train_data['CLAIM_FLAG']


# In[ ]:


X_test = test_data[['MVR_PTS', 'TRAVTIME']]
X_test = stats.add_constant(X_test, prepend=True)

Y_test = test_data['CLAIM_FLAG']


# In[ ]:


threshold = train_data['CLAIM_FLAG'].value_counts()[1] / len(train_data)
threshold


# In[ ]:


logit = stats.MNLogit(Y_train, X_train)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

predicted_probabilities = thisFit.predict(X_test)
predicted_probabilities


# In[ ]:


predictions = []
for i in predicted_probabilities[1]:
    if i >= threshold:
        predictions.append(1)
    else:
        predictions.append(0)
len(predictions)        


# In[ ]:


accuracy = metrics.accuracy_score(Y_test, predictions)
misclassification = 1 - accuracy
misclassification


# In[ ]:




