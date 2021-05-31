import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.neighbors as kNN
import sklearn.svm as svm
import statsmodels.api as sm

numpy.random.seed(seed = 20181107)

# Generate the first multivariate normal sample
m0 = [1.3, 1.3]
s0 = [1.0, 1.0]
corr0 = [[1.0, 0.3], [0.3, 1.0]]

cov0 = corr0 * numpy.outer(s0, s0)
print('Covariance Matrix 0:\n', cov0)

mvn0 = numpy.append(numpy.full((700,1), 0),
                    numpy.random.multivariate_normal(m0, cov0, 700),
                    axis = 1)

# Generate the second multivariate normal sample
m1 = [-1.3, -1.3]
s1 = [0.8, 0.8]
corr1 = [[1.0, -0.1], [-0.1, 1.0]]

cov1 = corr1 * numpy.outer(s1, s1)
print('Covariance Matrix 1:\n', cov1)

mvn1 = numpy.append(numpy.full((300,1), 1),
                    numpy.random.multivariate_normal(m1, cov1, 300),
                    axis = 1)

mvnData = numpy.append(mvn0, mvn1, axis = 0)

# Scatterplot without any prior knowledge of the grouping variable
plt.figure(figsize=(10,10))
plt.scatter(x = mvnData[:,1], y = mvnData[:,2])
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Fit the 2 cluster solution
trainData = pandas.DataFrame(mvnData[:,[1,2]], columns = ['x', 'y'])
kmeans = cluster.KMeans(n_clusters=2, random_state=20181107).fit(trainData)
centroid = kmeans.cluster_centers_
print('Centroids:\n', centroid)

trainData['Cluster ID'] = kmeans.labels_

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['Cluster ID'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 10)
plt.scatter(x = centroid[:,0], y = centroid[:,1], c = 'black', marker = 'X', s = 150)
plt.grid(True)
plt.title('2-KMeans Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Cluster ID', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Scatterplot that uses prior information of the grouping variable
carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = mvnData[mvnData[:,0] == i]
    plt.scatter(x = subData[:,1],
                y = subData[:,2], c = carray[i], label = i, s = 10)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try the 5-nearest neighbor classifier
trainData = pandas.DataFrame(mvnData[:,[1,2]], columns = ['x', 'y'])
neigh = kNN.KNeighborsClassifier(n_neighbors=5, algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, mvnData[:,0])

# See the classification result
print('Mean Accuracy = ', neigh.score(trainData, mvnData[:,0]))
trainData['_PredictedClass_'] = nbrs.predict(trainData)

kNN_Mean = trainData.groupby('_PredictedClass_').mean()

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 10)
plt.scatter(x = kNN_Mean['x'], y = kNN_Mean['y'], c = 'black', marker = 'X', s = 150)
plt.grid(True)
plt.title('5-Nearest Neighbors')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try the logistic classifier
xTrain = pandas.DataFrame(mvnData[:,[1,2]], columns = ['x', 'y'])
xTrain = sm.add_constant(xTrain, prepend=True)

yTrain = pandas.DataFrame(mvnData[:,0], columns = ['group']).astype('category')

logit_model = sm.MNLogit(yTrain, xTrain)
thisFit = logit_model.fit()
print(thisFit.summary())

y_predProb = thisFit.predict(xTrain)
y_predictClass = pandas.to_numeric(y_predProb.idxmax(axis=1))

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

logistic_Mean = trainData.groupby('_PredictedClass_').mean()
print(logistic_Mean)

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 10)
plt.scatter(x = logistic_Mean['x'], y = logistic_Mean['y'], c = 'black', marker = 'X', s = 150)
plt.grid(True)
plt.title('Multinomial Logistic')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try the sklearn.svm.LinearSVC
trainData = pandas.DataFrame(mvnData[:,[1,2]], columns = ['x', 'y'])
yTrain = pandas.DataFrame(mvnData[:,0], columns = ['group']).astype('category')

svm_Model = svm.LinearSVC(verbose = 1, random_state = 20181107, max_iter = 1000)
thisFit = svm_Model.fit(trainData, yTrain)

print('Intercept:\n', thisFit.intercept_)
print('Weight Coefficients:\n', thisFit.coef_)

y_predictClass = thisFit.predict(trainData)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 10)
plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'X', s = 150)
plt.plot([3.521094569, -4], [-2.370555172, 4], color = 'black', linestyle = ':')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

