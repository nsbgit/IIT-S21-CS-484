# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:44:05 2021

@author: pc
"""

import numpy as np

def EuclideanDistance (x, y):
    outDistance = np.sqrt(np.sum((x - y) ** 2))
    return outDistance

def ManhattanDistance (x, y):
    outDistance = np.sum(np.abs(x - y))
    return outDistance

def ChebyshevDistance (x, y):
    outDistance = np.max(np.abs(x - y))
    return outDistance

def CosineDistance (x, y):
    normX = np.sqrt(np.dot(x, x))
    normY = np.sqrt(np.dot(y, y))
    if normX > 0.0 and normY > 0.0:
        outDistance = 1.0 - np.dot(x, y) / normX / normY
    else:
        outDistance = np.nan
    return outDistance

clist = [-1.0, 1.0]
points = np.array([])
for u in clist:
    for v in clist:
        points = np.append(points, [u, v], axis = 0)
        
nPoints = len(clist) ** 2
points = np.reshape(points, (nPoints, 2))

distance_E = np.zeros((nPoints, nPoints))
distance_M = np.zeros((nPoints, nPoints))
distance_C = np.zeros((nPoints, nPoints))
distance_O = np.zeros((nPoints, nPoints))

for i in range(nPoints):
    for j in range(nPoints):
        distance_E[i, j] = EuclideanDistance(points[i, :], points[j, :])
        distance_M[i, j] = ManhattanDistance(points[i, :], points[j, :])
        distance_C[i, j] = ChebyshevDistance(points[i, :], points[j, :])
        distance_O[i, j] = CosineDistance(points[i, :], points[j, :])
        
print(distance_E)
print(distance_M)
print(distance_C)
print(distance_O)

