import numpy as np
import pandas as pd
import sklearn.svm as svm
#import statsmodels.api as sm
import sklearn.metrics as metrics
pd.options.mode.chained_assignment = None  # default='warn'

trainData = pd.read_csv('Q15.csv')


xTrain = trainData[['x','y']]
yTrain = trainData['group']
# svm_Model = svm.SVC(kernel = 'linear', random_state = 20191106, max_iter = -1)
# thisFit = svm_Model.fit(xTrain, yTrain)
# y_predictClass = thisFit.predict(xTrain)
# print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
# trainData['_PredictedClass_'] = y_predictClass
# svm_Mean = trainData.groupby('_PredictedClass_').mean()

# svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
#                     random_state = 20210325, max_iter = -1)
# thisFit = svm_Model.fit(xTrain, yTrain) 
# y_predictClass = thisFit.predict(xTrain)
# print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
# trainData['_PredictedClass_'] = y_predictClass
# print('Intercept = ', thisFit.intercept_)
# print('Coefficients = ', thisFit.coef_)
# print('Slope-Intercept form →',"{:.7f}".format(thisFit.intercept_[0]),
#       '+',"{:.7f}".format(thisFit.coef_[0][0]),'x +',"{:.7f}".format(thisFit.coef_[0][1]),
#       'y = 0\n')
# print('Accuracy Rate = ', metrics.accuracy_score(yTrain, y_predictClass),'\n')

#Build Support Vector Machine classifier
x_train = trainData[['x','y']]
y_train = trainData['group']
nObs = trainData.shape[0]
w_train = np.full(nObs, 1.0)
accuracy = np.zeros(50)
misclassification = np.zeros(50)
ensemblePredProb = np.zeros((nObs, 2))
for iter in range(50):
    svm_Model = svm.SVC(kernel = 'linear', random_state = 20191106, max_iter = -1)
    thisFit = svm_Model.fit(x_train, y_train,w_train)
    y_predictClass = thisFit.predict(x_train)
    accuracy[iter] = metrics.accuracy_score(y_train, y_predictClass,sample_weight = w_train)
    if (accuracy[iter] >= 0.9999999):
        break    
    # Update the weights
    eventError = np.where(y_train == 0, (1 - y_predictClass), (0 - y_predictClass))
    predClass = np.where(y_predictClass >= 0.1, 1, 0)
    w_train = np.where(predClass != y_train, 1 + np.abs(eventError), np.abs(eventError))

#print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
#trainData['_PredictedClass_'] = y_predictClass

#svm_Mean = trainData.groupby('_PredictedClass_').mean()
#print(svm_Mean)

#print('Intercept = ', thisFit.intercept_)
#print('Coefficients = ', thisFit.coef_)







# data['u'] = 2 * data['x'] + 1
# data['v'] = 4 * data['y'] + 2
# xTrain = data[['u','v']]
# yTrain = data['group']
# svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
#                     random_state = 20210325)
# thisFit = svm_Model.fit(xTrain, yTrain) 
# y_predictClass = thisFit.predict(xTrain)
# print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
# data['_PredictedClass_'] = y_predictClass
# print('Intercept = ', thisFit.intercept_)
# print('Coefficients = ', thisFit.coef_)
# print('Slope-Intercept form →',"{:.7f}".format(thisFit.intercept_[0]),
#       '+',"{:.7f}".format(thisFit.coef_[0][0]),'u +',"{:.7f}".format(thisFit.coef_[0][1]),
#       'v = 0\n')