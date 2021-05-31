# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 01:25:28 2018

@author: suhas
"""

import pandas
import numpy
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pandas.read_csv('policy_2001.csv',
                       delimiter=',')

print('Number of Observations = ', data.shape[0])
print(data.groupby('CLAIM_FLAG').size() / data.shape[0])


#dummies
credScoreBand = data[['CREDIT_SCORE_BAND']].astype('category')
hmeq = pandas.get_dummies(credScoreBand)
hmeq = hmeq.join(data[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF','TRAVTIME','CLAIM_FLAG']])


#test train split
hmeq_train, hmeq_test = train_test_split(hmeq, test_size = 0.3, random_state = 20181010)

print('Number of Observations in Training = ', hmeq_train.shape[0])
print('Number of Observations in Testing = ', hmeq_test.shape[0])

print(hmeq_train.groupby('CLAIM_FLAG').size() / hmeq_train.shape[0])

print(hmeq_test.groupby('CLAIM_FLAG').size() / hmeq_test.shape[0])


hmeq_train, hmeq_test = train_test_split(hmeq, test_size = 0.3, random_state = 20181010, stratify = hmeq['CLAIM_FLAG'])

print('Number of Observations in Training = ', hmeq_train.shape[0])
print('Number of Observations in Testing = ', hmeq_test.shape[0])

print(hmeq_train.groupby('CLAIM_FLAG').size() / hmeq_train.shape[0])

print(hmeq_test.groupby('CLAIM_FLAG').size() / hmeq_test.shape[0])

#nearest neighbours
trainData = hmeq_train[['CREDIT_SCORE_BAND_450 - 619','CREDIT_SCORE_BAND_620 - 659','CREDIT_SCORE_BAND_660 - 749','CREDIT_SCORE_BAND_750 +','BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF','TRAVTIME']]
target = hmeq_train['CLAIM_FLAG']

testData = hmeq_test[['CREDIT_SCORE_BAND_450 - 619','CREDIT_SCORE_BAND_620 - 659','CREDIT_SCORE_BAND_660 - 749','CREDIT_SCORE_BAND_750 +','BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF','TRAVTIME']]
target_test = hmeq_test['CLAIM_FLAG']

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)

# See the classification result
class_result = nbrs.predict(trainData)
#print(class_result)

# See the classification probabilities
class_prob = nbrs.predict_proba(trainData)
#print(class_prob)

accuracy = nbrs.score(testData, target_test)
print("Nearest neighbours accuracy:",accuracy)

#predicted probabilities for test partition
neigh_prob_test = nbrs.predict_proba(testData)
df_pred_prob_neigh = pandas.DataFrame(neigh_prob_test)

thresh_pred_knn = numpy.zeros(len(neigh_prob_test))
for i in range(len(neigh_prob_test)):
    if(neigh_prob_test[i][1] >= 0.287703):
        thresh_pred_knn[i] = 1
    else:
        thresh_pred_knn[i] = 0

thresh_accuracy_knn = metrics.accuracy_score(target_test, thresh_pred_knn)
missclassification_knn = 1 - thresh_accuracy_knn

knn_test_result = nbrs.predict(testData)
knn_auc = metrics.roc_auc_score(target_test, knn_test_result)

#classification tree
from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10,random_state = 20181010)
tree = classTree.fit(trainData, target)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(classTree.score(testData, target_test)))

#predicted probabilities for test partition
tree_prob_test = tree.predict_proba(testData)
df_pred_prob_tree = pandas.DataFrame(tree_prob_test)

thresh_pred_tree = numpy.zeros(len(tree_prob_test))
for i in range(len(tree_prob_test)):
    if(tree_prob_test[i][1] >= 0.287703):
        thresh_pred_tree[i] = 1
    else:
        thresh_pred_tree[i] = 0

thresh_accuracy_tree = metrics.accuracy_score(target_test, thresh_pred_tree)
missclassification_tree = 1 - thresh_accuracy_tree

tree_test_result = tree.predict(testData)
tree_auc = metrics.roc_auc_score(target_test, tree_test_result)

#logistic regression
import sklearn.metrics as metrics
import statsmodels.api as api


y = target
#y_category = y.cat.categories


X = trainData
X = api.add_constant(X, prepend=True)

logit = api.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

#y_predictClass = y_category[y_predict]
#
#y_confusion = metrics.confusion_matrix(y, y_predictClass)
#print("Confusion Matrix (Row is True, Column is Predicted) = \n")
#print(y_confusion)
#
#y_accuracy = metrics.accuracy_score(y, y_predictClass)
#print("Accuracy Score = ", y_accuracy)


y_accuracy = metrics.accuracy_score(y, y_predict)
print("Accuracy Score = ", y_accuracy)

#predicted probabilities for test partition
X_test = testData
X_test = api.add_constant(testData, prepend=True)
df_pred_prob_logistic = thisFit.predict(X_test)

X_test_predict = pandas.to_numeric(df_pred_prob_logistic.idxmax(axis=1))
X_test_accuracy = metrics.accuracy_score(target_test, X_test_predict)
print("Logistic Accuracy Score testing partition = ", X_test_accuracy)

thresh_pred_logistic = numpy.zeros(len(df_pred_prob_logistic))
for i in range(len(df_pred_prob_logistic)):
    if(df_pred_prob_logistic.iloc[i,1] >= 0.287703):
        thresh_pred_logistic[i] = 1
    else:
        thresh_pred_logistic[i] = 0

thresh_accuracy_logistic = metrics.accuracy_score(target_test, thresh_pred_logistic)
missclassification_logistic = 1 - thresh_accuracy_logistic


lr_auc = metrics.roc_auc_score(target_test, X_test_predict)

#lift chart

# Define a function to compute the coordinates of the Lift chart
def compute_lift_coordinates (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug = 'N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = numpy.percentile(EventPredProb, numpy.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = numpy.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb.iloc[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pandas.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN /totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                    axis = 1, ignore_index = True)
    LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                              1:'Decile %',
                                              2:'Gain N',
                                              3:'Gain %',
                                              4:'Response %',
                                              5:'Lift'}, axis = 'columns')

    # Construct the Accumulative Lift chart table
    accCountTable = countTable.cumsum(axis = 0)
    decileN = accCountTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = accCountTable[EventValue]
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    lift = responsePct / overallResponsePct

    accLiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                       axis = 1, ignore_index = True)
    accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                    1:'Acc. Decile %',
                                                    2:'Acc. Gain N',
                                                    3:'Acc. Gain %',
                                                    4:'Acc. Response %',
                                                    5:'Acc. Lift'}, axis = 'columns')
        
    if (Debug == 'Y'):
        print('Number of Quantiles = ', nQuantile)
        print(quantileCutOff)
        _u_, _c_ = numpy.unique(quantileIndex, return_counts = True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)
    
#endfunction
    
score_test = pandas.concat([target_test, df_pred_prob_logistic], axis = 1)

# Get the Lift chart coordinates for logistic
lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = score_test['CLAIM_FLAG'],
        EventValue = 1,
        EventPredProb = score_test[1],
        Debug = 'Y')

# Get the Lift chart coordinates for nearest neighbours
target_test.reset_index(drop=True, inplace=True)
df_pred_prob_neigh.reset_index(drop=True, inplace=True)
score_test_neigh = pandas.concat([target_test, df_pred_prob_neigh], axis = 1)
lift_coordinates_neigh, acc_lift_coordinates_neigh = compute_lift_coordinates (
        DepVar = score_test_neigh['CLAIM_FLAG'],
        EventValue = 1,
        EventPredProb = score_test_neigh[1],
        Debug = 'Y')

# Get the Lift chart coordinates for tree
score_test_tree = pandas.concat([target_test, df_pred_prob_tree], axis = 1)
lift_coordinates_tree, acc_lift_coordinates_tree = compute_lift_coordinates (
        DepVar = score_test_tree['CLAIM_FLAG'],
        EventValue = 1,
        EventPredProb = score_test_tree[1],
        Debug = 'Y')

# Draw the Lift chart
plt.plot(lift_coordinates.index, lift_coordinates['Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Testing Partition on Logistic")
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Lift")
plt.show()

# Draw the Accumulative Lift chart
plt.plot(acc_lift_coordinates.index, acc_lift_coordinates['Acc. Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(acc_lift_coordinates_neigh.index, acc_lift_coordinates_neigh['Acc. Lift'], marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(acc_lift_coordinates_tree.index, acc_lift_coordinates_tree['Acc. Lift'], marker = 'o',
         color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.legend(['Logistic Regression', 'KNN', 'Decision Tree'], loc='upper right')
plt.title("Accumulated Lift chart for each of the three models using the Testing partition")
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.show()


#roc
cutoffs=[-2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 2]
Sensitivity=[]
OneMinusSpecificity=[]

for c in cutoffs:
    predictions = (df_pred_prob_logistic.iloc[:,1] >= c).astype(int)
    confusion_matrix = metrics.confusion_matrix(target_test, predictions)
    sensitivity_val = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
    specificity_val = confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[0][0])
    Sensitivity.append(sensitivity_val)
    OneMinusSpecificity.append(specificity_val)
    
rocData = pandas.DataFrame({'CutOff': cutoffs,'Sensitivity': Sensitivity,'OneMinusSpecificity': OneMinusSpecificity})


#roc tree
cutoffs=[-2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 2]
sensitivity_tree=[]
OneMinusSpecificity_tree=[]

for c in cutoffs:
    predictions_tree = (df_pred_prob_tree.iloc[:,1] >= c).astype(int)
    confusion_matrix_tree = metrics.confusion_matrix(target_test, predictions_tree)
    sensitivity_val = confusion_matrix_tree[1][1] / (confusion_matrix_tree[1][0] + confusion_matrix_tree[1][1])
    specificity_val = confusion_matrix_tree[0][1] / (confusion_matrix_tree[0][1] + confusion_matrix_tree[0][0])
    sensitivity_tree.append(sensitivity_val)
    OneMinusSpecificity_tree.append(specificity_val)
    
rocData_tree = pandas.DataFrame({'CutOff': cutoffs,'sensitivity_tree': sensitivity_tree,'OneMinusSpecificity_tree': OneMinusSpecificity_tree})


#roc KNN
cutoffs=[-2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 2]
sensitivity_neigh=[]
OneMinusSpecificity_neigh=[]

for c in cutoffs:
    predictions_neigh = (df_pred_prob_neigh.iloc[:,1] >= c).astype(int)
    confusion_matrix_neigh = metrics.confusion_matrix(target_test, predictions_neigh)
    s=confusion_matrix_neigh[1][1] / (confusion_matrix_neigh[1][0] + confusion_matrix_neigh[1][1])
    sp=confusion_matrix_neigh[0][1] / (confusion_matrix_neigh[0][1] + confusion_matrix_neigh[0][0])
    sensitivity_neigh.append(s)
    OneMinusSpecificity_neigh.append(sp)
    
rocData_neigh = pandas.DataFrame({'CutOff': cutoffs,'sensitivity_neigh': sensitivity_neigh,'OneMinusSpecificity_neigh': OneMinusSpecificity_neigh})



ax = plt.gca()
ax.set_aspect('equal')
plt.figure(figsize=(6,6))
plt.plot(rocData['OneMinusSpecificity'], rocData['Sensitivity'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)

plt.plot(rocData_tree['OneMinusSpecificity_tree'], rocData_tree['sensitivity_tree'], marker = 'o',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)

plt.plot(rocData_neigh['OneMinusSpecificity_neigh'], rocData_neigh['sensitivity_neigh'], marker = 'o',
         color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)

plt.legend(['Logistic Regression', 'Decision tree', 'KNN'], loc='upper left')

plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.title("ROC curve for each of the three models using the Testing partition")
plt.show()