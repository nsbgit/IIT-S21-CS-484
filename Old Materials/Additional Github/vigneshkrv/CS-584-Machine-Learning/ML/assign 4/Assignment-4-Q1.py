#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import sympy
import scipy
import sklearn.metrics as metrics
import statsmodels.api as sm
data = pd.read_csv('Purchase_Likelihood.csv')



# In[40]:


data.head()


# In[41]:


def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = sm.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


# In[42]:


data = data.dropna()


# In[43]:


y = data['A'].astype('category')

significanceList = []
dof = []
chi_sq_stats = []
# group_size, homeowner, married_couple, group_size * homeowner, and homeowner * married_couple 
# Specify JOB and REASON as categorical variables
xI = pd.get_dummies(data[['group_size']].astype('category'))
xJ = pd.get_dummies(data[['homeowner']].astype('category'))
xK = pd.get_dummies(data[['married_couple']].astype('category'))


# In[44]:


# Intercept only
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')


# In[45]:


# Intercept + group_size
designX = sm.add_constant(xI, prepend=True)
LLK_1R, DF_1R, fullParams_1R = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R - LLK0)
testDF = DF_1R - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
chi_sq_stats.append(testDev)
dof.append(testDF)
significanceList.append(testPValue)


# In[46]:


# Intercept + group_size + homeowner
designX = xI
designX = designX.join(xJ)
designX = sm.add_constant(designX, prepend=True)
LLK_1R_1J, DF_1R_1J, fullParams_1R_1J = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J - LLK_1R)
testDF = DF_1R_1J - DF_1R
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
chi_sq_stats.append(testDev)
dof.append(testDF)
significanceList.append(testPValue)


# In[47]:


# Intercept + group_size + homeowner + married_couple
designX = xI
designX = designX.join(xJ)
designX = designX.join(xK)
designX = sm.add_constant(designX, prepend=True)
LLK_1R_1J, DF_1R_1J, fullParams_1R_1J = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J - LLK_1R)
testDF = DF_1R_1J - DF_1R
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
chi_sq_stats.append(testDev)
dof.append(testDF)
significanceList.append(testPValue)


# In[48]:


# Intercept + group_size + homeowner + married_couple + group_size * homeowner
designX = xI
designX = designX.join(xJ)
designX = designX.join(xK)
# Create the columns for the group_size * homeowner interaction effect
xIJ = create_interaction(xI, xJ)
designX = designX.join(xIJ)
# xJK = create_interaction(xJ,xK)
designX = sm.add_constant(designX, prepend=True)
LLK_2RJ, DF_2RJ, fullParams_2RJ = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJ - LLK_1R_1J)
testDF = DF_2RJ - DF_1R_1J
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
chi_sq_stats.append(testDev)
dof.append(testDF)
significanceList.append(testPValue)


# In[49]:


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple
designX = xI
designX = designX.join(xJ)
designX = designX.join(xK)

# Create the columns for the group_size * homeowner interaction effect
xIJ = create_interaction(xI, xJ)
designX = designX.join(xIJ)
xJK = create_interaction(xJ,xK)
designX = designX.join(xJK)

designX = sm.add_constant(designX, prepend=True)
LLK_2RJ, DF_2RJ, fullParams_2RJ = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJ - LLK_1R_1J)
testDF = DF_2RJ - DF_1R_1J
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
chi_sq_stats.append(testDev)
dof.append(testDF)
significanceList.append(testPValue)


# In[50]:


# ALl except => (0, 1, 2, 3, 5, 7, 9, 11, 13, 17) => alias parameters
# Degree of Freedom => 8
# c) chi sq test stats , signifcance , degree of freedom
# d) table values for feature importance index


import math
index_vals = [
              'Intercept + group_size',
              'Intercept + group_size + homeowner',
              'Intercept + group_size + homeowner + married_couple',
              'Intercept + group_size + homeowner + married_couple + group_size * homeowner',
              'Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple']

logs = [-math.log(s,10) if s!=0 else 0 for s in significanceList]
d = {'Chi Square Test':chi_sq_stats,'Significance':significanceList,'Degrees Of Freedom':dof,'Feature Importance Index':logs}
dataframe = pd.DataFrame(d,index=index_vals)
dataframe


# In[51]:


# (0, 1, 2, 3, 5, 7, 9, 11, 13, 17)
X = [0,1, 2, 3, 5, 7,9, 11, 13, 17]
designX.loc[:, ~designX.columns.isin([list(designX.columns)[i] for i in range(0,len(designX)) if i in X])].head()
# designX.head()


# In[76]:


import itertools
# e)
combination = []
group_size = data['group_size'].unique().tolist()
group_size.sort()
combination.append(group_size)
combination.append(data['homeowner'].unique().tolist())
combination.append(data['married_couple'].unique().tolist())
print('Combinations list', combination)
X = list(itertools.product(*combination))
combination = []
for i in X:
    combination.append(i)
    
df = pd.DataFrame(combination,columns=['group_size','homeowner','married_couple'])#index=[i for i in range(1,17)])
df


# In[61]:


len(designX.columns)


# In[62]:


df


# In[106]:


# Create the columns for the group_size * homeowner interaction effect
x1 = pd.get_dummies(df[['group_size']].astype('category'))
x2 = pd.get_dummies(df[['homeowner']].astype('category'))
x3 = pd.get_dummies(df[['married_couple']].astype('category'))

y = data['A'].astype('category')
y_ = pd.DataFrame(y.where(y.isnull(), 1))


t = x1.join(x2)
t = t.join(x3)
t = t.join(create_interaction(x1,x2))
t = t.join(create_interaction(x2,x3))
t = t.join(y_)
# print(t.shape)
# print(t)
# x12 = create_interaction(x1, x2)

logit = sm.MNLogit(y, designX)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
pred = thisFit.predict(t.astype('float'))
# pd.DataFrame(pred)


# In[108]:


pd.DataFrame(pred) #e)


# In[98]:


t


# In[ ]:




