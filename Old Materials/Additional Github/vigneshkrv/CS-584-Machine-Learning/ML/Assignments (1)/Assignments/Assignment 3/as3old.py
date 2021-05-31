# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:09:35 2018

@author: suhas
"""

import statsmodels.api as st
import pandas as pd
import numpy as np

df = pd.read_csv('Purchase_Likelihood.csv')


dfPredictors = df[['group_size','homeowner','married_couple']]

dfTarget = df[['A']]

mdl = st.MNLogit(dfTarget, dfPredictors)
print(mdl.exog_names)
 
mdl_fit = mdl.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

parameters = mdl_fit.params



print(mdl_fit.summary())
print(mdl_fit.mle_retvals)
print(parameters)
print(mdl_fit.predict(exog = [2,1,1]))
print(mdl.loglike(parameters.values))