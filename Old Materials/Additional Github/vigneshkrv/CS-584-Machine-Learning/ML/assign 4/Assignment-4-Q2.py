#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn.naive_bayes as naive_bayes
import itertools


data = pd.read_csv('Purchase_Likelihood.csv')
data.head()


# In[2]:


# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

    countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
    print("Frequency Table: \n", countTable)
    print( )

#     if (show == 'ROW' or show == 'BOTH'):
#         rowFraction = countTable.div(countTable.sum(1), axis='index')
#         print("Row Fraction Table: \n", rowFraction)
#         print( )

#     if (show == 'COLUMN' or show == 'BOTH'):
#         columnFraction = countTable.div(countTable.sum(0), axis='columns')
#         print("Column Fraction Table: \n", columnFraction)
#         print( )

    return countTable


# In[3]:


feature = ['group_size','homeowner','married_couple']
target = 'A'

df = []
data = data.dropna()
for pred in feature:
    df.append(RowWithColumn(rowVar = data[target], columnVar = data[pred], show = 'BOTH'))
    
data[feature] = 2 - data[feature]

xTrain = data[feature].astype('category')
yTrain = data[target].astype('category')

_objNB = naive_bayes.BernoulliNB(alpha = 0)
thisFit = _objNB.fit(xTrain, yTrain)

print('Probability of each class')
print(np.exp(thisFit.class_log_prior_))

print('Empirical probability of features given a class, P(x_i|y)')
print(np.exp(thisFit.feature_log_prob_))

print('Number of samples encountered for each class during fitting')
print(thisFit.class_count_)

print('Number of samples encountered for each (class, feature) during fitting')
print(thisFit.feature_count_)

# yTrain_predProb = _objNB.predict_proba(xTrain)

# # Create the all possible combinations of the features' values
# xTest = pd.DataFrame(list(itertools.product([0,1], repeat = len(feature))), columns = feature)

# # Score the xTest and append the predicted probabilities to the xTest
# yTest_predProb = pd.DataFrame(_objNB.predict_proba(xTest))
# yTest_score = pd.concat([xTest, yTest_predProb], axis = 1)

# print(yTest_predProb)
# print(yTest_score)


# In[4]:


df[0]


# In[5]:


df[1]


# In[13]:


df[2]


# In[8]:


# Define a function that performs the Chi-square test
import scipy
def cramerVFunc (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable   # debugging flag (Y/N) 
    debug = 'Y'     
):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))
    
    if (debug == 'Y'):
        print('Observed Count:\n', obsCount)
        print('Column Total:\n', cTotal)
        print('Row Total:\n', rTotal)
        print('Overall Total:\n', nTotal)
        print('Expected Count:\n', expCount)
        print('\n')

       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return (cramerV)


# In[12]:


for i in list(df[0].columns.values):
    print('For Feature {} '.format(i),cramerVFunc(df[0][i],df[0].index.values))
print()
for i in list(df[1].columns.values):
    print('For Feature {} '.format(i),cramerVFunc(df[1][i],df[1].index.values))
print()    
for i in list(df[2].columns.values):
    print('For Feature {} '.format(i),cramerVFunc(df[2][i],df[2].index.values))


# In[ ]:





# In[ ]:




