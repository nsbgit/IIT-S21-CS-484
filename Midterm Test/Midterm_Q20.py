# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 12:06:56 2021

@author: pc
"""

import pandas as pd

data = pd.read_csv('policy_2001.csv', usecols=['CLAIM_FLAG','CREDIT_SCORE_BAND', 'BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME'])
dummies = pd.get_dummies(data['CREDIT_SCORE_BAND'])
data = data.drop(columns=['CREDIT_SCORE_BAND'])
data = data.join(dummies)

