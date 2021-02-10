# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:23:38 2020

@author: Lipi
"""

import pandas as pd
import numpy as np
import statsmodels.api as stats
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


train_data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\FinalExam\WineQuality_Train.csv')
test_data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\FinalExam\WineQuality_Test.csv')
threshold = 0.1961733010776

Xtrain = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Xtrain = stats.add_constant(Xtrain, prepend=True)
Ytrain = train_data['quality_grp']

Xtest = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Xtest = stats.add_constant(Xtest, prepend=True)
Ytest = test_data['quality_grp']

#MULTINOMIAL MODEL
logit = stats.MNLogit(Ytrain, Xtrain)
thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)

pred_prob = thisFit.predict(Xtest)

predictions = []
count = 0
for i in range(len(pred_prob)):
    if pred_prob[1][i] >= threshold:
        predictions.append(1)
    else:
        predictions.append(0)
        
for k in range(len(Ytest)):
    if Ytest[k] != predictions[k]:
        count += 1
        
MNL_misclassification = count / len(Ytest)
print('The misclassification rate for the Multinomial Logistic model =', MNL_misclassification)


#Support Vector Machine
Xtrain = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Ytrain = train_data['quality_grp']

Xtest = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
Ytest = test_data['quality_grp']


svm_model = SVC(kernel='linear', random_state=20200428, max_iter=-1)
thisFit = svm_model.fit(Xtrain,Ytrain)
predictions = thisFit.predict(Xtest)

SVM_misclassification = 1 - accuracy_score(predictions, Ytest)
print('The misclassification rate for SVM model =', SVM_misclassification)

#Multi-Layer Perceptor
def calculate_misclassification(pred_prob):
    predictions = []
    count = 0
    
    for i in pred_prob:
        if i[1] >= threshold:
            predictions.append(1)
        else:
            predictions.append(0)
            
    for k in range(len(Ytest)):
        if Ytest[k] != predictions[k]:
            count += 1
     
    return count / len(Ytest) 


def build_nerural_netwotk(nHiddenNeuron, nLayer):

    NN = MLPClassifier(hidden_layer_sizes=((nHiddenNeuron,)*nLayer), activation='relu', verbose=False, solver='lbfgs', 
                        learning_rate_init=0.1, max_iter=5000, random_state=20200428) 

    thisFit = NN.fit(Xtrain,Ytrain)
    pred_prob = NN.predict_proba(Xtrain)
    misclassification = calculate_misclassification(pred_prob)
    loss = NN.loss_
    
    return loss, misclassification



result = pd.DataFrame(columns=['nLayer', 'nHiddenNeuron', 'loss', 'missclassification'])

for nLayer in np.arange(1,11):
    for nHiddenNeuron in np.arange(5,11):
        loss, misclassification = build_nerural_netwotk(nHiddenNeuron, nLayer)
        result = result.append(pd.DataFrame([[nLayer, nHiddenNeuron, loss,misclassification]],
                                                columns=['nLayer', 'nHiddenNeuron', 'loss', 'missclassification']))
        

sorted_result = result.sort_values(by='loss')
sorted_result


nHiddenNeuron = 10 
nLayer = 9
NN = MLPClassifier(hidden_layer_sizes=((nHiddenNeuron,)*nLayer), activation='relu', verbose=False, solver='lbfgs', 
                        learning_rate_init=0.1, max_iter=5000, random_state=20200428)

thisFit = NN.fit(Xtest,Ytest)
predict_prob = NN.predict_proba(Xtest)
MLP_misclassification = calculate_misclassification(predict_prob)
print('The classification rate for Multi-layer perceptron model =', MLP_misclassification)



#### Question 7 #################################################33

yib = [22.4673913043478,
29.5044247787611,
25.0363636363636,]
yib = np.array(yib)

ni = [92,226,110]
ni = np.array(ni)
yb = 26.8434579439252

ans7 = sum(np.square(yb-yib)*ni) / 14074.5116822429000
yib
