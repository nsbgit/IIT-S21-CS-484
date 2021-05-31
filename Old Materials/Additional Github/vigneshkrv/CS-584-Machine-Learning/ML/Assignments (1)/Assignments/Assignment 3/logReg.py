# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:06:55 2018

@author: suhas
"""

import statsmodels.api as api
import pandas as pd
import numpy as np

df = pd.read_csv('Purchase_Likelihood.csv')
A = df['A'].astype('category')
y = A
y_category = y.cat.categories

group_size = df[['group_size']].astype('category')
X = pd.get_dummies(group_size)
X = X.join(df[['homeowner', 'married_couple']])
X = api.add_constant(X, prepend=True)

logit = api.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params
print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))
print(thisFit.summary())
print("\n Prediction : ", thisFit.predict(exog = [1,1,0,0,0,1,1]))
print("\n Prediction : ", thisFit.predict(exog = [1,1,0,0,0,0,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,1,0,0,0,1,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,1,0,0,0,0,1]))

print("\n Prediction : ", thisFit.predict(exog = [1,0,1,0,0,1,1]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,1,0,0,0,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,1,0,0,1,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,1,0,0,0,1]))

print("\n Prediction : ", thisFit.predict(exog = [1,0,0,1,0,1,1]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,0,1,0,0,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,0,1,0,1,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,0,1,0,0,1]))

print("\n Prediction : ", thisFit.predict(exog = [1,0,0,0,1,1,1]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,0,0,1,0,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,0,0,1,1,0]))
print("\n Prediction : ", thisFit.predict(exog = [1,0,0,0,1,0,1]))