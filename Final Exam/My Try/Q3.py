# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:20:16 2021

@author: pc
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# Initialize data of lists
data = {'x1':[0, 0.4, 0.7, 0.5, 0.5, 0.6, 0.3, 0.1, 0.8, 0.8]
        ,'x2':[0.6, 0.4, 0.8, 0.2, 0.8, 0, 0.2, 0.6, 0.8, 0]
        ,'y':[-0.6, -0.6, 0.6, 1.8, 1.2, 1.2, 1.4, 0.6, 1.8, 1.6]}

# using dictionary to convert specific columns
convert_dict = {'x1': float,
                'x2': float,
                'y': float
               }

# create data
df = pd.DataFrame(data)
df = df.astype(convert_dict)
print(df.dtypes)

#Specify the data
x = df[['x1', 'x2']]
y = df['y']

# Find optimal number of neighbours
result = pd.DataFrame()
max_neighbors = x.shape[0]

for k in range(max_neighbors):
    kNNSpec = KNeighborsRegressor(n_neighbors = (k+1), metric = 'chebyshev')
    nbrs = kNNSpec.fit(x, y)
    pred_y = nbrs.predict(x)
    error_y = y - pred_y
    sse_y = np.sum(np.absolute(error_y))
    result = result.append([[(k+1), sse_y]], ignore_index = True)
 
result = result.rename(columns = {0: 'Number of Neighbors', 1: 'Sum of Squared Error'})

plt.scatter(result['Number of Neighbors'], result['Sum of Squared Error'])
plt.xlabel('Number of Neighbors')
plt.ylabel('Sum of Squared Error')
plt.xticks(np.arange(1,max_neighbors+1,1))
plt.grid(axis = 'both')
plt.show()

suggested_neighbor = result.nsmallest(2, 'Sum of Squared Error').tail(1).reset_index(drop=True).loc[0]['Number of Neighbors']

print(f'The number of neighbors that yields the smallest criterion is k = {suggested_neighbor}')