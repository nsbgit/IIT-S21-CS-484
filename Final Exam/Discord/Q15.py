import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.svm as svm

trainData = pandas.read_csv('Q15.csv')

# Scatterplot that uses prior information of the grouping variable

# Build Support Vector Machine classifier
xTrain = trainData[['x','y']]
yTrain = trainData['group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
xx = numpy.linspace(-6, 6)
yy = numpy.zeros((len(xx),3))
for j in range(1):
    w = thisFit.coef_[j,:]
    a = -w[0] / w[1]
    yy[:,j] = a * xx - (thisFit.intercept_[j]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green', 'blue']
plt.figure(figsize=(10,10))
for i in range(3):
    subData = trainData[trainData['_PredictedClass_'] == (i+1)]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = (i+1), s = 25)

# Convert to the polar coordinates
trainData['radius'] = numpy.sqrt(trainData['x']**2 + trainData['y']**2)
trainData['theta'] = numpy.arctan2(trainData['y'], trainData['x'])

def customArcTan (z):
    theta = numpy.where(z < 0.0, 2.0*numpy.pi+z, z)
    return (theta)

trainData['theta'] = trainData['theta'].apply(customArcTan)

# Build Support Vector Machine classifier
xTrain = trainData[['radius','theta']]
yTrain = trainData['group']

print(xTrain.isnull().sum())

xTrain = trainData[['radius','theta']]
yTrain = trainData['group']

svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)
