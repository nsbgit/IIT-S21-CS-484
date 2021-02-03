# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:58:21 2021

@author: Sukanta Sharma
"""

import numpy as np

def CosineD (x, y):
    normX = np.sqrt(np.dot(x, x))
    normY = np.sqrt(np.dot(y, y))
    if normX > 0.0 and normY > 0.0:
        outDistance = 1.0 - np.dot(x, y) / normX / normY
    else:
        outDistance = np.nan
    return outDistance

# data
X = np.array([[2, 1, 0, 0, 0, 0],
              [2, 0, 1, 0, 0, 0],
              [1, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 1]])

# probe
P = np.array([[3, 0, 0, 0, 1, 1],
              [0, 1, 1, 1, 0, 0]])

cosine_D = np.zeros((4, 2))
for i in range(4):
    for j in range(2):
        cosine_D[i, j] = CosineD(X[i, :], P[j, :])
   
print("\nCosine Distance")
print(cosine_D)

minInColumns = np.amin(cosine_D, axis=0)

minIndex = np.where(cosine_D == np.amin(cosine_D, axis=1))

print("\nMinimum Distance")
print(minInColumns)

print("\nIndex of Minimum Distance")
print(minIndex)