#!/usr/bin/env python
# coding: utf-8

# #  Assignment 4 : Machine Learning Question 2

# In[48]:


import pandas as pd
import numpy as np
import scipy
import statsmodels.api as stats

df = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week10_assignment4\\Purchase_Likelihood.csv')
print(df.head())


# In[102]:


def RowWithColumn (
    rowVar,          # Row variable
    columnVar,       # Column predictor
    show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

    countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
    print("Frequency Table: \n", countTable)
    print( )

    if (show == 'ROW' or show == 'BOTH'):
        rowFraction = countTable.div(countTable.sum(1), axis='index')
        print("Row Fraction Table: \n", rowFraction)
        print( )

    if (show == 'COLUMN' or show == 'BOTH'):
        columnFraction = countTable.div(countTable.sum(0), axis='columns')
        print("Column Fraction Table: \n", columnFraction)
        print( )

    return
# Specify the roles
feature = ['group_size', 'homeowner', 'married_couple']
target = 'insurance'

df = df.dropna()

# Look at the row distribution

print(df.groupby(target).size())
freq = df.groupby('insurance').size()
table = pd.DataFrame(columns = ['count', 'class probability'])
table['count'] = freq
table['class probability'] = table['count']/df.shape[0]
table

for pred in feature:
    RowWithColumn(rowVar = df[target], columnVar = df[pred], show = 'ROW')


# In[103]:


def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))

    # Debugging codes omitted to enhance readability
       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)


# In[101]:


data_count = df.groupby('insurance').count()['group_size']
data_prop = data_count / df.shape[0]
data_grouped = pd.DataFrame({'insurance': data_count.index, 
                                    'Count': data_count.values, 
                                    'Class probabilities of target variable': data_prop.values})
data_grouped


# In[52]:


catPred = ['group_size', 'homeowner', 'married_couple']



testResult = pd.DataFrame(index = catPred ,
                              columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in catPred:
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(df[pred], df['insurance'], debug = 'Y')
    testResult.loc[pred] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]
    

rankSig = testResult.sort_values('Significance', axis = 0, ascending = True)

rankAssoc = testResult.sort_values('Measure', axis = 0, ascending = False)

#print(rankSig)
print(rankAssoc)


# In[53]:


data_group_size = pd.crosstab(df.insurance, df.group_size, margins = False, dropna = False)

data_homeowner = pd.crosstab(df.insurance, df.homeowner, margins = False, dropna = False)

data_married_couple = pd.crosstab(df.insurance, df.married_couple, margins = False, dropna = False)


# In[54]:


def get_valid_probabilities(predictors):
    cond_prob_0 = ((data_grouped['Count'][0] / data_grouped['Count'].sum()) * 
                   (data_group_size[predictors[0]][0] / data_group_size.loc[[0]].sum(axis=1)[0]) * 
                   (data_homeowner[predictors[1]][0] / data_homeowner.loc[[0]].sum(axis=1)[0]) * 
                   (data_married_couple[predictors[2]][0] / data_married_couple.loc[[0]].sum(axis=1)[0]))
    cond_prob_1 = ((data_grouped['Count'][1] / data_grouped['Count'].sum()) * 
                   (data_group_size[predictors[0]][1] / data_group_size.loc[[1]].sum(axis=1)[1]) * 
                   (data_homeowner[predictors[1]][1] / data_homeowner.loc[[1]].sum(axis=1)[1]) * 
                   (data_married_couple[predictors[2]][1] / data_married_couple.loc[[1]].sum(axis=1)[1]))
    cond_prob_2 = ((data_grouped['Count'][2] / data_grouped['Count'].sum()) * 
                   (data_group_size[predictors[0]][2] / data_group_size.loc[[2]].sum(axis=1)[2]) * 
                   (data_homeowner[predictors[1]][2] / data_homeowner.loc[[2]].sum(axis=1)[2]) * 
                   (data_married_couple[predictors[2]][2] / data_married_couple.loc[[2]].sum(axis=1)[2]))
    sum_cond_probs = cond_prob_0 + cond_prob_1 + cond_prob_2
    valid_prob_0 = cond_prob_0 / sum_cond_probs
    valid_prob_1 = cond_prob_1 / sum_cond_probs
    valid_prob_2 = cond_prob_2 / sum_cond_probs

    return [valid_prob_0, valid_prob_1, valid_prob_2]


# In[55]:


group_sizes = sorted(list(df.group_size.unique()))
homeowners = sorted(list(df.homeowner.unique()))
married_couples = sorted(list(df.married_couple.unique()))


# In[56]:


import itertools
combinations = list(itertools.product(group_sizes, homeowners, married_couples))


# In[82]:


naive_bayes_probabilities = []
for combination in combinations:
    temp = [get_valid_probabilities(combination)]
    naive_bayes_probabilities.extend(temp)
naive_bayes_probabilities


# In[104]:


df1=pd.DataFrame(naive_bayes_probabilities)
df2=pd.DataFrame(combinations)
result= pd.merge(df2,df1, left_index=True,right_index=True)
result1=result.rename(columns={'0_x':'Group_size','1_x':'Homeowner','2_x':'married_couple','0_y':'A=0','1_y':'A=1','2_y':'A=2'})
result1

pd_result1 = pd.DataFrame(result1)

# In[79]:


maximum=[]
for i in range(len(naive_bayes_probabilities)):
    temp=naive_bayes_probabilities[i][1]/naive_bayes_probabilities[i][0]
    maximum.append([temp])    
print(np.array(maximum).max())


# In[59]:


max_val = np.array(maximum).max()
index = np.where(maximum == max_val)[0][0]


# In[60]:


print("The maximum value occurs when group_size, homeowner, married_couple values are: ",combinations[index])
print("The maximum value is: ", max_val)


# In[ ]:





# In[ ]:





# In[ ]:




