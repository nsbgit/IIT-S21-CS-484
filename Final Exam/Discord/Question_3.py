import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("Q3.csv", header = 0)

# Specify the data
X = data[['x1', 'x2']]
y = data['y']

# Build nearest neighbors
result = pd.DataFrame()
for k in range(10):
   kNNSpec = KNeighborsRegressor(n_neighbors = (k+1), metric = 'euclidean')
   nbrs = kNNSpec.fit(X, y)
   pred_y = nbrs.predict(X)
   error_y = y - pred_y
   sse_y = np.sum(error_y ** 2)
   result = result.append([[(k+1), sse_y]], ignore_index = True)
 
result = result.rename(columns = {0: 'Number of Neighbors', 1: 'Sum of Squared Error'})

plt.scatter(result['Number of Neighbors'], result['Sum of Squared Error'])
plt.xlabel('Number of Neighbors')
plt.ylabel('Sum of Squared Error')
plt.xticks(np.arange(1,11,1))
plt.grid(axis = 'both')
plt.show()

print('The number of neighbors that yields the smallest criterion is k = 2')


