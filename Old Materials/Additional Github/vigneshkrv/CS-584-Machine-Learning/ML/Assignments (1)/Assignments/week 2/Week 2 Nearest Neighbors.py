# Load the PANDAS library
import pandas
cars = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\cars.csv',
                       delimiter=',')

# Examine a portion of the data frame
print(cars)

# Put the descriptive statistics into another dataframe
cars_description = cars.groupby('Origin').describe()

from sklearn.neighbors import NearestNeighbors as kNN
import numpy as np

kNNSpec = kNN(n_neighbors = 4, algorithm = 'brute', metric = 'euclidean')

trainData = cars[['Invoice', 'Horsepower', 'Weight']]
trainData.describe()

# Build nearest neighbors
nbrs = kNNSpec.fit(trainData)
distances, indices = nbrs.kneighbors(trainData)

# Find the nearest neighbors of these focal observations       
focal = [[173560, 477, 3131],     # Porsche_335
         [119600, 493, 4473],     # Mercedes-Benz_263
         [117854, 493, 4429],     # Mercedes-Benz_272
         [113388, 493, 4235]]     # Mercedes-Benz_271

myNeighbors = nbrs.kneighbors(focal, return_distance = False)
print("My Neighbors = \n", myNeighbors)

# Perform classification
target = cars[['Origin']]

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)

# See the classification result
class_result = nbrs.predict(trainData)
print(class_result)

# See the classification probabilities
class_prob = nbrs.predict_proba(trainData)
print(class_prob)

accuracy = nbrs.score(trainData, target)