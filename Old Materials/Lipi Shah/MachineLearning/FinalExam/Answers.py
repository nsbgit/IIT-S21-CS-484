# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:37:57 2020

@author: Lipi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

def BinWidth(Interquartile, n):
     bin_width = 2 * Interquartile*(n ** (-1/3))
     return bin_width
 
    
# Driver Function 
if __name__=='__main__': 
    data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\FinalExam\\FinalQ1.csv',
                       delimiter=',')
    x = data.iloc[:,0].values 
    n = len(x) 
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bin_width =  BinWidth(iqr, n)
    print ("**********************Question 1 *********************************************")
    print ("**********************Answer 1.a *********************************************")
    print("answer 1.a : According to Izenman (1991) method, the recomemded bin width for Histogram x is ", bin_width)
    print ("**********************Answer 1.b *********************************************")
    print("answer 1.b minimum value of the field x is: ",min(x)," and maximum value of the field x is :", max(x))
    a = math.floor(min(x))
    b = math.ceil(max(x))
    print ("**********************Answer 1.c *********************************************")
    print("answer 1.c the largest integer less than the minimum value of the field x is", a , "and smallest integer greater than the maximum value of the field x is", b)
    #question 1.d
    
    print ("**********************Answer 1.d *********************************************")
    h= 4.0
    N = len(x)
    a = 74
    m = a + h/2
    m_values = []
    p = []
    d = dict()
    counter = 0
    while (m <= b):
        m_values.append(m)
        w_u = []
        
        u = (x-m)/h
        for i in u:
            if i> -1/2 and i < 1/2:
                w_u.append(1)
            else:
                w_u.append(0)
        p.append(sum(w_u)/(N * h))
        print("[",m_values[counter],':', p[counter],"]",end = " ")
        counter = counter + 1
        m = m + h
     
    h = 4
    num_of_bin = (b-a)/h
    sns.distplot(x, bins=int(num_of_bin),kde=True,color='blue')
    plt.title("Histogram of Density Estimater for attribute x at h = 4.0")
    plt.xlabel("Attribute x values")
    plt.ylabel("Density Estimator")
    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.show()