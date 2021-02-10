import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics

import sklearn.svm as svm

trainData = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week12_assignment5\\SpiralWithCluster.csv')

y_threshold = trainData['SpectralCluster'].mean()


#########################################Question 2 a ########################################

# Build Support Vector Machine classifier
xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20191108, max_iter = -1, 
                    decision_function_shape  = 'ovr')
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = numpy.linspace(-3, 3)
yy = a * xx - (thisFit.intercept_[0]) / w[1]

#########################################Question 2 b ########################################

misclassification  = 1 - metrics.accuracy_score(yTrain, y_predictClass)
print("Misclassification :", misclassification)


#########################################Question 2 c ########################################

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'x', s = 100)
plt.plot(xx, yy, color = 'black', linestyle = '-')
plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
#plt.scatter(cc[:,0], cc[:,1], color = 'black', marker = '+', s = 100)
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#########################################Question 2 d ########################################

# Convert to the polar coordinates
trainData['radius'] = numpy.sqrt(trainData['x']**2 + trainData['y']**2)
trainData['theta'] = numpy.arctan2(trainData['y'], trainData['x'])

def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

trainData['theta'] = trainData['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = trainData[['radius','theta']]
yTrain = trainData['SpectralCluster']

print(xTrain.isnull().sum())

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191108, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(0, 6)
yy = numpy.zeros((len(xx),3))
for j in range(1):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]


# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == (i+1)]
    plt.scatter(x = subData['radius'],
                y = subData['theta'], c = carray[i], label = (i+1), s = 25)
plt.plot(xx, yy[:,0], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,1], color = 'black', linestyle = '-')
plt.plot(xx, yy[:,2], color = 'black', linestyle = '-')
plt.grid(True)
plt.title('Support Vector Machines on Three Segments')
plt.xlabel('Radius')
plt.ylabel('Angle')
plt.ylim(-0.5, 6.5)
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

