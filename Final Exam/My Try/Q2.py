# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:46:53 2021

@author: pc
"""

import numpy as np
from scipy.spatial import distance

def CosineD (x, y):
    normX = np.sqrt(np.dot(x, x))
    normY = np.sqrt(np.dot(y, y))
    if normX > 0.0 and normY > 0.0:
        outDistance = 1.0 - np.dot(x, y) / normX / normY
    else:
        outDistance = np.nan
    return outDistance

# Question 2
# given focal point & points (0,0), (4,0), (0,4), (4,4)
# we will find the cosine distance between (2,2) and the rest of these points!

focal_point = np.array([2, 2])
data_points = np.array([[0, 0]
                        ,[0,4]
                        ,[4,0]
                        ,[4,4]])

cosine_distance = np.empty(shape=(len(data_points), 1))
cosine_distance[:] = np.nan
# cosine_distance = []

for i in range(len(data_points)):
    data_point = data_points[i]
    cosine_distance[i] = distance.cosine(focal_point, data_point)
    

# (2,2) & (0,0)
x = np.array([2,2])
y = np.array([0,0])
print('Cosine distance btw (2,2) & (0,0) = ',distance.cosine(x, y))
print('Cosine distance btw (2,2) & (0,0) = ',CosineD(x, y))
# (2,2) & (4,0)
x = np.array([2,2])
y = np.array([4,0])
print('Cosine distance btw (2,2) & (4,0) = ',distance.cosine(x, y))
print('Cosine distance btw (2,2) & (4,0) = ',CosineD(x, y))
# (2,2) & (0,4)
x = np.array([2,2])
y = np.array([0,4])
print('Cosine distance btw (2,2) & (0,4) = ',distance.cosine(x, y))
print('Cosine distance btw (2,2) & (0,4) = ',CosineD(x, y))
# (2,2) & (4,4)
x = np.array([2,2])
y = np.array([4,4])
print('Cosine distance btw (2,2) & (4,4) = ',distance.cosine(x, y))
print('Cosine distance btw (2,2) & (4,4) = ',CosineD(x, y))
print('shortest distance is for (2,2) & (4,4)!')
print(np.min(cosine_distance))