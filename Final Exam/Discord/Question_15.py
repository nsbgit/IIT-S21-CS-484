import pandas as pd
import sklearn.svm as svm
import sklearn.metrics as metrics
pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv('Q15.csv')

xTrain = data[['x','y']]
yTrain = data['group']
svm_Model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr',
                    random_state = 20210325)
thisFit = svm_Model.fit(xTrain, yTrain) 
y_predictClass = thisFit.predict(xTrain)
print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
data['_PredictedClass_'] = y_predictClass
print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)
print('Slope-Intercept form →',"{:.7f}".format(thisFit.intercept_[0]),
      '+',"{:.7f}".format(thisFit.coef_[0][0]),'x +',"{:.7f}".format(thisFit.coef_[0][1]),
      'y = 0\n')

