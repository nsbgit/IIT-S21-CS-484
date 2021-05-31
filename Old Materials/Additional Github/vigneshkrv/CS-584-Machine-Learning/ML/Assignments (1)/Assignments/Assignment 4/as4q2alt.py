# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:39:12 2018

@author: suhas
"""

import numpy
import pandas
import sklearn.naive_bayes as NB

dataFrame = pandas.read_csv('Purchase_Likelihood.csv',delimiter=',')
class_p = [0.215996,0.640462,0.143542]
gs_p = pandas.crosstab(index = dataFrame.A, columns = dataFrame.group_size, normalize = 'index')

home_p = pandas.crosstab(index = dataFrame.A, columns = dataFrame.homeowner, normalize = 'index')

married_p = pandas.crosstab(index = dataFrame.A, columns = dataFrame.married_couple, normalize = 'index')


def emperical_nb(class_p, grp_size, homeown, married_coup):
    cond_prob0 = class_p[0] * grp_size[0] * homeown[0] * married_coup[0]
    cond_prob1 = class_p[1] * grp_size[1] * homeown[1] * married_coup[1]
    cond_prob2 = class_p[2] * grp_size[2] * homeown[2] * married_coup[2]
    total = cond_prob0 + cond_prob1 + cond_prob2
    
    prob0 = cond_prob0/total
    prob1 = cond_prob1/total
    prob2 = cond_prob2/total
    
    return (prob0,prob1,prob2)
    

gs=[1,2,3,4]
ho=[0,1]
mc=[0,1]
prob_1=[]


for g in gs:
    for h in ho:
        for m in mc:
            print('group_size=',g,' homeowner=',h,' married_couple=',m)
            p = (emperical_nb(class_p,gs_p[g],home_p[h],married_p[m]))
            print(p)
            prob_1.append((g,h,m,p))

sorted(prob_1, key = lambda prob_1: prob_1[3][1], reverse = True)