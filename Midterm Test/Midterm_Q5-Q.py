# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:26:33 2021

@author: pc
"""

import math
import numpy


def ChebyshevDistance (x, y):
    outDistance = numpy.max(numpy.abs(x - y))
    return outDistance

def getEntropy(x):
    total = numpy.sum(x)
    pi = x / total
    el = pi * numpy.log2(pi)
    entropy = numpy.sum(el)
    return -1 * entropy

# ------------ Q.5----------------------------------
print("Q.5:")
# number of unique items in the universal set I
m = 100
# k-itemset, where 1<= k <= m
k = 5
print('We can possibly generarte {0} {1}-itemset.'.format((math.factorial(m) / (math.factorial(k) * math.factorial(m - k))), k))

# ------------ Q.9----------------------------------
print("Q.9:")
giniFrequency = numpy.array([262, 1007, 1662, 1510, 559])
prob = numpy.divide(giniFrequency, numpy.sum(giniFrequency))
prob = numpy.square(prob)
giniIndexValue = 1 - numpy.sum(prob)
print("Gini Index Value of the root node is: {0}".format(giniIndexValue))

# ------------ Q.10----------------------------------
print("Q.10:")
pi_iJ = 1662 / 5000
pi_ij = 559 / 5000
b_j = math.log(pi_ij / pi_iJ)
print("The estimated intercept of category V is : {0}".format(b_j))

# ------------ Q.11----------------------------------
print("Q.11:")
cluster0Centroids = numpy.array([6.34, 6.82, 7.21, 7.18, 7.47])
cluster1Centroids = numpy.array([8.04, 8.56, 9.42, 8.08, 7.70])
observation = numpy.array([9.7, 10.7, 11.4, 7.8, 6.5])
dC0 = ChebyshevDistance(cluster0Centroids, observation)
dC1 = ChebyshevDistance(cluster1Centroids, observation)

# ------------ Q.13----------------------------------
print("Q.13:")
x = [123,518,972,968,403]
entropy = getEntropy(x)
print(entropy)

