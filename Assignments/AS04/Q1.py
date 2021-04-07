# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:33:35 2021

@author: Sukanta Sharma
"""

import numpy
import pandas as pd
import scipy.stats as sdist
import sklearn.naive_bayes as naive_bayes


# Define a function that performs the Pearson Chi-square test
#   xCat - Input categorical feature (array-like or Series)
#   yCat - Input categorical target field (array-like or Series)

def PearsonChiSquareTest(xCat, yCat):
    # Generate the crosstabulation
    obsCount = pd.crosstab(index=xCat, columns=yCat, margins=False, dropna=True)
    xNCat = obsCount.shape[0]
    yNCat = obsCount.shape[1]
    cTotal = obsCount.sum(axis=1)
    rTotal = obsCount.sum(axis=0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    # Calculate the Chi-Square statistics
    chiSqStat = ((obsCount - expCount) ** 2 / expCount).to_numpy().sum()
    chiSqDf = (xNCat - 1) * (yNCat - 1)
    if (chiSqDf > 0):
        chiSqSig = sdist.chi2.sf(chiSqStat, chiSqDf)
    else:
        chiSqSig = numpy.NaN

    # Calculate Cramers'V
    cramersV = chiSqStat / nTotal
    if rTotal.size < cTotal.size:
        cramersV = cramersV / (rTotal.size - 1.0)
    else:
        cramersV = cramersV / (cTotal.size - 1.0)

    cramersV = numpy.sqrt(cramersV)

    return (xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig, cramersV)


def CrossTabulate(rowVar, columnVar):
    cross_table = pd.crosstab(index=[rowVar], columns=[columnVar])  # , margins = False, dropna = True)
    # cross_table = pd.crosstab(index=data['LE_Split'], columns=data.iloc[:, 1], margins=True, dropna=True)
    print(cross_table.transpose())


# ----------------------------------------------------------------------------
#                           Q1
# ----------------------------------------------------------------------------

data = pd.read_csv('Purchase_Likelihood.csv')
data = data.dropna()

# ----------------------------------------------------------------------------
#                           Q1.a
# ----------------------------------------------------------------------------
frequency = data.groupby('insurance').size()
frequencyPr = pd.DataFrame(columns=['freq', 'prob'])
frequencyPr['freq'] = frequency
frequencyPr['prob'] = frequencyPr['freq'] / data.shape[0]
print(frequencyPr.transpose())
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q1.b
# ----------------------------------------------------------------------------
CrossTabulate(rowVar=data['insurance'], columnVar=data['group_size'])
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q1.c
# ----------------------------------------------------------------------------
CrossTabulate(rowVar=data['insurance'], columnVar=data['homeowner'])
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q1.d
# ----------------------------------------------------------------------------
CrossTabulate(rowVar=data['insurance'], columnVar=data['married_couple'])
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q1.e
# ----------------------------------------------------------------------------
cat_pred = ['group_size', 'homeowner', 'married_couple']
testResult = pd.DataFrame()
for pred in cat_pred:
    xNCat, yNCat, chiSqStat, chiSqDf, chiSqSig, cramerV = PearsonChiSquareTest(data[pred],data['insurance'])
    testResult = testResult.append([[pred, cramerV]], ignore_index=True)
print("Cramer's V:\n")
print(testResult)
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q1.f
# ----------------------------------------------------------------------------
xTrain = data[cat_pred].astype('category')
yTrain = data['insurance'].astype('category')

_objNB = naive_bayes.MultinomialNB(alpha=1.0e-10)
thisFit = _objNB.fit(xTrain, yTrain)
groupSize_data = [1, 2, 3, 4]
homeowner_data = [0, 1]
marriedCouple_data = [0, 1]
insurance_data = [0, 1, 2]

x_data = []

for i in groupSize_data:
    for j in homeowner_data:
        for k in marriedCouple_data:
            x_data.append([i, j, k])

x_test = pd.DataFrame(x_data, columns=cat_pred)
x_test = x_test[cat_pred].astype('category')
y_test = pd.DataFrame(_objNB.predict_proba(x_test),columns=['Prob (insurance = 0)', 'Prob (insurance = 1)', 'Prob (insurance = 2)'])
result = pd.concat([x_test, y_test], axis=1)
print(result)
# result.to_csv('Q1fResult.csv', index=False)
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q1.g
# ----------------------------------------------------------------------------
result['Odd Value(Prob (insurance = 1)/Prob (insurance = 2))'] = result['Prob (insurance = 1)'] / result['Prob (insurance = 2)']
print(result[['group_size', 'homeowner', 'married_couple', 'Odd Value(Prob (insurance = 1)/Prob (insurance = 2))']])
# result.to_csv('QgfResult.csv', index=False)
print('-+' * 40, end='\n')
print(result.loc[result['Odd Value(Prob (insurance = 1)/Prob (insurance = 2))'].idxmax()])
# result.loc[result['Odd Value(Prob (insurance = 1)/Prob (insurance = 2))'].idxmax()].to_csv('Q1g_2Result.csv', index=True)
print('-*' * 40, end='\n')