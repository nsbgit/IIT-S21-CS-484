# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:33:35 2021

@author: Sukanta Sharma
"""

import numpy
import pandas as pd
# pd.options.display.max_columns = 1000
import scipy.stats as sdist
# import scipy
# import statsmodels.api as stats
import sklearn.naive_bayes as naive_bayes


# Define a function that performs the Pearson Chi-square test
#   xCat - Input categorical feature (array-like or Series)
#   yCat - Input categorical target field (array-like or Series)

def PearsonChiSquareTest (xCat, yCat):
    # Generate the crosstabulation
    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    xNCat = obsCount.shape[0]
    yNCat = obsCount.shape[1]
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    # Calculate the Chi-Square statistics
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (xNCat - 1) * (yNCat - 1)
    if (chiSqDf > 0):
       chiSqSig = sdist.chi2.sf(chiSqStat, chiSqDf)
    else:
       chiSqSig = numpy.NaN
       
    # Calculate Cramers'V
    cramerV = chiSqStat / nTotal
    if rTotal.size < cTotal.size:
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    
    cramerV = numpy.sqrt(cramerV)

    return (xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig, cramerV)


def CrossTabulate(rowVar, columnVar):
    countTable = pd.crosstab(index = [rowVar], columns = [columnVar])#, margins = False, dropna = True)
    # cross_table = pd.crosstab(index=data['LE_Split'], columns=data.iloc[:, 1], margins=True, dropna=True)
    # print('Frequency Table:')
    # print(countTable)
    print(countTable.transpose())

# ----------------------------------------------------------------------------
#                           Q1
# ----------------------------------------------------------------------------

purchasell = pd.read_csv('Purchase_Likelihood.csv')
purchasell = purchasell.dropna()


# ----------------------------------------------------------------------------
#                           Q1.a
# ----------------------------------------------------------------------------
frequen = purchasell.groupby('insurance').size()
atable = pd.DataFrame(columns=['freq','prob'])
atable['freq']=frequen
atable['prob']=atable['freq']/purchasell.shape[0]
print(atable.transpose())
print('-'*40,end='\n')

# ----------------------------------------------------------------------------
#                           Q1.b
# ----------------------------------------------------------------------------
CrossTabulate(rowVar=purchasell['insurance'], columnVar=purchasell['group_size'])
print('-'*40,end='\n')


# ----------------------------------------------------------------------------
#                           Q1.c
# ----------------------------------------------------------------------------
CrossTabulate(rowVar=purchasell['insurance'], columnVar=purchasell['homeowner'])
print('-'*40,end='\n')


# ----------------------------------------------------------------------------
#                           Q1.d
# ----------------------------------------------------------------------------
CrossTabulate(rowVar=purchasell['insurance'], columnVar=purchasell['married_couple'])
print('-'*40,end='\n')


# ----------------------------------------------------------------------------
#                           Q1.e
# ----------------------------------------------------------------------------
cat_pred = ['group_size', 'homeowner', 'married_couple']
testResult = pd.DataFrame()
for pred in cat_pred:
    xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig, cramerV = PearsonChiSquareTest(purchasell[pred], purchasell['insurance'])
    testResult = testResult.append([[pred, cramerV]], ignore_index = True)
print("Cramer's V:\n")
print(testResult)
print('-'*40,end='\n')


# ----------------------------------------------------------------------------
#                           Q1.f
# ----------------------------------------------------------------------------
xTrain = purchasell[cat_pred].astype('category')
yTrain = purchasell['insurance'].astype('category')

_objNB = naive_bayes.MultinomialNB(alpha = 1.0e-10)
thisFit = _objNB.fit(xTrain, yTrain)
gs_d = [1, 2, 3, 4]
ho_d = [0, 1]
mc_d = [0, 1]
insurance_d = [0, 1, 2]

x_data = []

for gsd in gs_d:
    for hod in ho_d:
        for mcd in mc_d:
            x_data.append([gsd, hod, mcd])

x_test = pd.DataFrame(x_data, columns=cat_pred)
x_test = x_test[cat_pred].astype('category')
y_test_pred_prob = pd.DataFrame(_objNB.predict_proba(x_test), columns=['Prob (insurance = 0)', 'Prob (insurance = 1)', 'Prob (insurance = 2)'])
y_test_score = pd.concat([x_test, y_test_pred_prob], axis=1)
print(y_test_score)
y_test_score.to_csv('Q1fResult.csv', index=False)
print('-'*40,end='\n')


# ----------------------------------------------------------------------------
#                           Q1.g
# ----------------------------------------------------------------------------
y_test_score['Odd Value(Prob (insurance = 1)/Prob (insurance = 2))'] = y_test_score['Prob (insurance = 1)'] / y_test_score['Prob (insurance = 2)']
print(y_test_score[['group_size','homeowner','married_couple','Odd Value(Prob (insurance = 1)/Prob (insurance = 2))']])
y_test_score.to_csv('QgfResult.csv', index=False)
print('-+'*40,end='\n')
print(y_test_score.loc[y_test_score['Odd Value(Prob (insurance = 1)/Prob (insurance = 2))'].idxmax()])   
y_test_score.loc[y_test_score['Odd Value(Prob (insurance = 1)/Prob (insurance = 2))'].idxmax()].to_csv('Q1g_2Result.csv', index=True)
print('-*'*40,end='\n')   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    