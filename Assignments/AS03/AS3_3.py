# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 12:20:31 2021

@author: pc
"""

import numpy
import pandas
import scipy

import statsmodels.api as stats
import statsmodels.formula.api as smf

predictorList = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
droplist = ['y']
def doTest(LLK1, DF1, step):
    predList = predictorList.copy()
    cols = ['Index','Model Form','Number of Free Parameters','Log-Likelihood','Deviance','Degrees of Freedom','Chi-Square Significance','AIC','BIC']
    df = pandas.DataFrame(columns=cols)
    i = 0
    for predictor in predList:
        droplist2 = droplist.copy()
        droplist2.append(predictor)
        X = cars.drop(columns=droplist2)
        X = stats.add_constant(X, prepend=True)
        DF0 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)
        
        logit = stats.MNLogit(y, X)
        thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
        thisParameter = thisFit.params
        LLK0 = logit.loglike(thisParameter.values)
        
        Deviance = 2 * (LLK1 - LLK0)
        DF = DF1 - DF0
        pValue = scipy.stats.chi2.sf(Deviance, DF)
        
        # print(thisFit.summary2())
        # print("\n\nFor {0}:".format(predictor))
        # print("Model Log-Likelihood Value =", LLK0)
        # print("Number of Free Parameters =", DF0)
        # print("Deviance (Statistic, DF, Significance)", Deviance, DF, pValue)
        # print(type(i))
        i = i + 1
        indx = '{0}.{1}'.format(step, i)
        model = 'Intercept '
        modelList = predList.copy()
        modelList.remove(predictor)
        for e in modelList:
            model += '+ {0} '.format(e)
        data = {'Index': indx,
                'Model Form':model,
                'Number of Free Parameters':DF0,
                'Log-Likelihood':LLK0,
                'Deviance':Deviance,
                'Degrees of Freedom':DF,
                'Chi-Square Significance':pValue,
                'AIC':thisFit.aic,
                'BIC':thisFit.bic}
        df = df.append(data, ignore_index=True)
        
    # df.to_csv('out.csv',index=False)
    return df


def removePredictor(pred):
    global DF1, LLK1, dfm, df
    predictorList.remove(pred)
    droplist.append(pred)
    X = cars.drop(columns=droplist)
    X = stats.add_constant(X, prepend=True)
    DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)
    
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK1 = logit.loglike(thisParameter.values)
    
        
        


cars = pandas.read_csv('sample_v10.csv')


# ---------------- 3.a ----------------------------
frequencyTable = cars['y'].value_counts()
print('Frequency table of the categorical target field is:\n')
print(frequencyTable)


# --------------3.b -------------------------

# model = smf.mnlogit('y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10', cars)
# fii = model.fit()
# print(fii.summary())
# print('\n\n')
# print(fii.summary2())

nObs = cars.shape[0]

# Specify y as a categorical variable
Origin = cars['y'].astype('category')
y = Origin
y_category = y.cat.categories


# Backward Selection
# Consider Model 1 y = x1 + ... + x10
X = cars.drop(columns=['y'])
X = stats.add_constant(X, prepend=True)
DF1 = numpy.linalg.matrix_rank(X) * (len(y_category) - 1)

logit = stats.MNLogit(y, X)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
LLK1 = logit.loglike(thisParameter.values)

print(thisFit.summary())
print("Model Log-Likelihood Value =", LLK1)
print("Number of Free Parameters =", DF1)


df = doTest(LLK1, DF1,1)
dfm = df.copy()
removePredictor('x7')
df = doTest(LLK1, DF1,2)
dfm = dfm.append(df, ignore_index=True)
removePredictor('x3')
df = doTest(LLK1, DF1,3)
dfm = dfm.append(df, ignore_index=True)
removePredictor('x2')
df = doTest(LLK1, DF1,4)
dfm = dfm.append(df, ignore_index=True)
removePredictor('x5')
df = doTest(LLK1, DF1,5)
dfm = dfm.append(df, ignore_index=True)
removePredictor('x9')
df = doTest(LLK1, DF1,6)
dfm = dfm.append(df, ignore_index=True)
removePredictor('x6')
df = doTest(LLK1, DF1,7)
dfm = dfm.append(df, ignore_index=True)
removePredictor('x8')
df = doTest(LLK1, DF1,8)
dfm = dfm.append(df, ignore_index=True)
dfm.to_csv('out.csv',index=False)
