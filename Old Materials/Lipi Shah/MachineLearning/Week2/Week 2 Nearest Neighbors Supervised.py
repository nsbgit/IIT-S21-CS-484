# Load the PANDAS library
import pandas
from sklearn.neighbors import KNeighborsClassifier
import numpy

cars = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\Week2\\cars.csv',
                       delimiter=',')

cars["CaseID"] = cars["Make"] + "_" + cars.index.values.astype(str)

cars_wIndex = cars.set_index("CaseID")

trainData = cars_wIndex[['Invoice', 'Horsepower', 'Weight']]

# Perform classification
# Specify target: 0 = Asia, 1 = Europe, 2 = USA
target = cars_wIndex['Origin']

neigh = KNeighborsClassifier(n_neighbors=4 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)

# See the classification probabilities
class_prob = nbrs.predict_proba(trainData)
print(class_prob)

# Calculate the Misclassification Rate
targetClass = ['Asia', 'Europe', 'USA']

nMissClass = 0
for i in range(cars_wIndex.shape[0]):
    j = numpy.argmax(class_prob[i][:])
    predictClass = targetClass[j]
    if (predictClass != target.iloc[i]):
        nMissClass += 1

print(nMissClass)

rateMissClass = nMissClass / cars_wIndex.shape[0]
print('Misclassification Rate = ', rateMissClass)