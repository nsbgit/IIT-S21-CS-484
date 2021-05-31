# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
import sklearn.naive_bayes as NB
import statsmodels.api as api

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

Threshold = 0.20469083

def ModelMetrics (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Threshold):      # The probability threshold for declaring a predicted event

    # The Area Under Curve metric
    AUC = metrics.roc_auc_score(DepVar, EventPredProb)

    # The Root Average Squared Error and the Misclassification Rate
    nObs = len(DepVar)
    RASE = 0
    MisClassRate = 0
    for i in range(nObs):
        p = EventPredProb[i]
        if (DepVar[i] == EventValue):
            RASE += (1.0 - p)**2
            if (p < Threshold):
                MisClassRate += 1
        else:
            RASE += p**2
            if (p >= Threshold):
                MisClassRate += 1
    RASE = np.sqrt(RASE / nObs)
    MisClassRate /= nObs

    return(AUC, RASE, MisClassRate)
    
    
# Define a function to compute the coordinates of the Lift chart
def compute_lift_coordinates (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug = 'N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = np.percentile(EventPredProb, np.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = np.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pd.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN /totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
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

    accLiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
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
        _u_, _c_ = np.unique(quantileIndex, return_counts = True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)
    
    
#def classifier_decisionTree(train_data, test_data, train_label, test_label):
#    clf = DecisionTreeClassifier(max_depth=10, random_state=243)
#    print("Training Decision Trees")
#    clf.fit(train_data, train_label)
#    pred_label = clf.predict(test_data)
#    testTreePredProb = clf.predict_proba(test_data)
#        
#    testTreeAUC, testTreeRASE, testTreeMisClassRate = ModelMetrics (test_label, 1, testTreePredProb[:,1], Threshold )
#    print('Decision Tree')
#    print('          Area Under Curve = {:.6f}' .format(testTreeAUC))
#    print('Root Average Squared Error = {:.6f}' .format(testTreeRASE))
#    print('    Misclassification Rate = {:.6f}' .format(testTreeMisClassRate))
#  
#    dt_accuracy = metrics.accuracy_score(test_label , pred_label )
#    dt_misclassification = 1 - dt_accuracy
#    print("dt_misclassification = ",dt_misclassification)
#        
#    return pred_label

train = pd.read_csv('fleet_train.csv')
train = train.drop(['record_id', 'Measurement_timestamp','fleetid','truckid'], axis=1)
test = pd.read_csv('fleet_monitor_notscored_2.csv')
test = test.drop(['record_id', 'Measurement_timestamp','fleetid','truckid','period'], axis=1)

#for train data
train_region_1=train.loc[train['Region']==1]
train_region_1= train_region_1.drop(['Region'],axis =1)

train_region_2=train.loc[train['Region']==2]
train_region_2= train_region_2.drop(['Region'],axis =1)

train_region_3=train.loc[train['Region']==3]
train_region_3= train_region_3.drop(['Region'],axis =1)

#for test data
test_region_1=test.loc[test['Region']==1]
test_region_1= test_region_1.drop(['Region'],axis =1)

test_region_2=test.loc[test['Region']==2]
test_region_2= test_region_2.drop(['Region'],axis =1)

test_region_3=test.loc[test['Region']==3]
test_region_3= test_region_3.drop(['Region'],axis =1)

#spliting into x,y

#train_region_1_x = train_region_1.iloc[:, 1:].values
#train_region_1_y = train_region_1.iloc[:, 0].values
#
#train_region_2_x = train_region_2.iloc[:, 1:].values
#train_region_2_y = train_region_2.iloc[:, 0].values
#
#train_region_3_x = train_region_3.iloc[:, 1:].values
#train_region_3_y = train_region_3.iloc[:, 0].values
#
#
#
#test_region_1_x = test_region_1.iloc[:, 1:].values
#test_region_1_y = test_region_1.iloc[:, 0].values
#
#test_region_2_x = test_region_2.iloc[:, 1:].values
#test_region_2_y = test_region_2.iloc[:, 0].values
#
#test_region_3_x = test_region_3.iloc[:, 1:].values
#test_region_3_y = test_region_3.iloc[:, 0].values

train_region_1_x = train_region_1.iloc[:, 1:]
train_region_1_y = train_region_1.iloc[:, 0]

train_region_2_x = train_region_2.iloc[:, 1:]
train_region_2_y = train_region_2.iloc[:, 0]

train_region_3_x = train_region_3.iloc[:, 1:]
train_region_3_y = train_region_3.iloc[:, 0]

#intercepts
train_region_1_x_int = api.add_constant(train_region_1_x, prepend=True)
train_region_2_x_int = api.add_constant(train_region_2_x, prepend=True) 
train_region_3_x_int = api.add_constant(train_region_3_x, prepend=True)



test_region_1_x = test_region_1.iloc[:, 1:]
test_region_1_y = test_region_1.iloc[:, 0]

test_region_2_x = test_region_2.iloc[:, 1:]
test_region_2_y = test_region_2.iloc[:, 0]

test_region_3_x = test_region_3.iloc[:, 1:] 
test_region_3_y = test_region_3.iloc[:, 0]

#intercepts
test_region_1_x_int = api.add_constant(test_region_1_x, prepend=True)
test_region_2_x_int = api.add_constant(test_region_2_x, prepend=True) 
test_region_3_x_int = api.add_constant(test_region_3_x, prepend=True)

# training the classifier

#classifier_decisionTree(train_region_1_x,test_region_1_x,train_region_1_y,test_region_1_y)
#classifier_decisionTree(train_region_3_x,test_region_3_x,train_region_3_y,test_region_3_y)

import graphviz
from sklearn import tree
#all decision tree
clf = DecisionTreeClassifier(max_depth=6, criterion = 'entropy',random_state=0)
#print("Training Decision Trees")
#clf.fit(train_region_1_x, train_region_1_y)
#pred_label = clf.predict(test_region_1_x)
#PredProbregion1 = clf.predict_proba(test_region_1_x)
#
clf.fit(train_region_2_x, train_region_2_y)
pred_label = clf.predict(test_region_2_x)
PredProbregion2 = clf.predict_proba(test_region_2_x)

y_accuracy = metrics.accuracy_score(test_region_2_y, pred_label, normalize = True)
print("Accuracy Score 2= ", y_accuracy)

X_name = list(train_region_2_x)
dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_name,
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
print(graph)
graph.render('tree2')


clf.fit(train_region_3_x, train_region_3_y)
pred_label = clf.predict(test_region_3_x)
PredProbregion3 = clf.predict_proba(test_region_3_x)

y_accuracy = metrics.accuracy_score(test_region_3_y, pred_label, normalize = True)
print("Accuracy Score 3= ", y_accuracy)

X_name = list(train_region_3_x)
dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_name,
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
print(graph)
graph.render('tree3')


#all NB
#classifier = NB.GaussianNB().fit(train_region_1_x, train_region_1_y)
#pred_label = classifier.predict(test_region_1_x)
#PredProbregion1 = classifier.predict_proba(test_region_1_x)
#
#classifier = NB.GaussianNB().fit(train_region_2_x, train_region_2_y)
#pred_label = classifier.predict(test_region_2_x)
#PredProbregion2 = classifier.predict_proba(test_region_2_x)
#
#classifier = NB.GaussianNB().fit(train_region_3_x, train_region_3_y)
#pred_label = classifier.predict(test_region_3_x)
#PredProbregion3 = classifier.predict_proba(test_region_3_x)

#logistic sklearn
model = LogisticRegression()
rfe = RFE(model,9)
fit = rfe.fit(train_region_1_x, train_region_1_y)
print("Num Features: ",fit.n_features_)
print("Selected Features: ",fit.support_)
skjd = fit.support_
print("Feature Ranking: ",fit.ranking_)
print("parameters: ", rfe.get_params(deep=True))
logpred = fit.predict(test_region_1_x)
PredProbregion1 = rfe.predict_proba(test_region_1_x)
#for idx, col_name in enumerate(X_train.columns):
print("params",rfe.estimator_.coef_)
y_accuracy = metrics.accuracy_score(test_region_1_y, logpred, normalize = True)
print("Accuracy Score logistic regression= ", y_accuracy)

#all logistic
#logit = api.MNLogit(train_region_1_y, train_region_1_x_int)
#thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
#PredProbregion1 = thisFit.predict(test_region_1_x_int)
#
#Origin = test_region_1_y.astype('category')
#y = Origin
#y_category = y.cat.categories
#y_predict = pd.to_numeric(PredProbregion1.idxmax(axis=1))
#y_predictClass = y_category[y_predict]
#y_accuracy = metrics.accuracy_score(y, y_predictClass)
#print("Accuracy Score = ", y_accuracy)
#print("Model Parameter Estimates:\n", thisFit.params)
#thisParameter = thisFit.params
#print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

#logit = api.MNLogit(train_region_2_y, train_region_2_x_int)
#thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
#PredProbregion2 = thisFit.predict(test_region_2_x_int)

#logit = api.MNLogit(train_region_3_y, train_region_3_x_int)
#thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
#PredProbregion3 = thisFit.predict(test_region_3_x_int)

#PredProbregion1 = PredProbregion1.values
#PredProbregion1[:,1] = 0.2
#PredProbregion1[:,0] = 0.8
#
#PredProbregion2[:,1] = 1/3
#PredProbregion2[:,0] = 2/3
#
#PredProbregion3[:,1] = 1/50
#PredProbregion3[:,0] = 49/50

testTreePredProb = np.concatenate((PredProbregion1, PredProbregion2, PredProbregion3))
test_s = np.concatenate((test_region_1_y, test_region_2_y, test_region_3_y))

    
testTreeAUC, testTreeRASE, testTreeMisClassRate = ModelMetrics (test_s, 1, testTreePredProb[:,1], Threshold )
testTreeFP, testTreeTP, testTreeThresholds = metrics.roc_curve(test_s, testTreePredProb[:,1], pos_label = 1)
testTreeLift, testTreeAccLift = compute_lift_coordinates (test_s, 1, testTreePredProb[:,1])


#    
#testTreeAUC, testTreeRASE, testTreeMisClassRate = ModelMetrics (test_region_1_y, 1, testTreePredProb[:,1], Threshold )
#testTreeFP, testTreeTP, testTreeThresholds = metrics.roc_curve(test_region_1_y, testTreePredProb[:,1], pos_label = 1)
#testTreeLift, testTreeAccLift = compute_lift_coordinates (test_region_1_y, 1, testTreePredProb[:,1])

print('Decision Tree')
print('          Area Under Curve = {:.6f}' .format(testTreeAUC))
print('Root Average Squared Error = {:.6f}' .format(testTreeRASE))
print('    Misclassification Rate = {:.6f}' .format(testTreeMisClassRate))

#dt_accuracy = metrics.accuracy_score(test_region_1_y , pred_label)
#dt_misclassification = 1 - dt_accuracy
#print("dt_misclassification = ",dt_misclassification)

plt.plot(testTreeFP, testTreeTP, marker = 'o',label = '3 models',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.title("Receiver Operating Characteristic Curve")
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis(aspect = 'equal')
plt.legend()
plt.show()

plt.plot(testTreeLift.index, testTreeLift['Lift'], marker = 'o', label = '3 models',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Lift Chart")
plt.grid(True)
plt.xticks(np.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Lift")
plt.legend()
plt.show()

plt.plot(testTreeAccLift.index, testTreeAccLift['Acc. Lift'], marker = 'o',label = '3 models',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("Accumulative Lift Chart")
plt.grid(True)
plt.xticks(np.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.legend()
plt.show()
