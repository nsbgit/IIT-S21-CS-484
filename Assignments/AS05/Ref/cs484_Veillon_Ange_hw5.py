# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 02:30:19 2021

@author: Ange
"""
import pandas as pd 
import numpy
from sklearn import metrics
import sklearn.tree as tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import random
random.seed(20210415)

train = pd.read_csv("C:/Users/Ange/Downloads/WineQuality_Train.csv").drop(['type','quality','pH','density','total_sulfur_dioxide','chlorides','volatile_acidity','fixed_acidity'],axis=1)
test = pd.read_csv("C:/Users/Ange/Downloads/WineQuality_Test.csv").drop(['type','quality','pH','density','total_sulfur_dioxide','chlorides','volatile_acidity','fixed_acidity'],axis=1)


x_train = train.drop('quality_grp',axis=1)
y_train = train['quality_grp']

x_test = test.drop('quality_grp',axis=1)
y_test = test['quality_grp']

print(type(x_train))
print(type(y_train))
def question_1():
    num_int=50
    nObs = train.shape[0]
    
    
    #part a
    w_train = numpy.full(nObs, 1.0)
    accuracy = numpy.zeros(num_int)
    ensemblePredProb = numpy.zeros((nObs, 2))
    
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415 )
    treeFit = classTree.fit(x_train, y_train)
    treePredProb = classTree.predict_proba(x_test)
    accuracy = classTree.score(x_test, y_test)
    print('Accuracy = ', accuracy)
    
    dot_data = tree.export_graphviz(treeFit,out_file=None,impurity = True, filled = True,feature_names = x_train.columns,class_names = ['0', '1'])
    
    graph = graphviz.Source(dot_data)
    graph
    
    #part b
    w_train = numpy.full(nObs, 1.0)
    accuracy = numpy.zeros(num_int)
    ensemblePredProb = numpy.zeros((nObs, 2))
    for iter in range(num_int):
        classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
        treeFit = classTree.fit(x_train, y_train, w_train)
        treePredProb = classTree.predict_proba(x_train)
        accuracy[iter] = classTree.score(x_train, y_train, w_train)
        ensemblePredProb += accuracy[iter] * treePredProb
    
        if (abs(1.0 - accuracy[iter]) < 0.0000001):
            break
        
        # Update the weights
        eventError = numpy.where(y_train == 1, (1 - treePredProb[:,1]), (treePredProb[:,1]))
        predClass = numpy.where(treePredProb[:,1] >= 0.2, 1, 0)
        w_train = numpy.where(predClass != y_train, 2+numpy.abs(eventError), numpy.abs(eventError))
        if iter==0:
            print('Accuracy at iter 1 = ', accuracy[iter])
    # Calculate the final predicted probabilities
    ensemblePredProb /= numpy.sum(accuracy)
    
    train['predCluster'] = numpy.where(ensemblePredProb[:,1] >= 0.2, 1, 0)
    
    misclass=[1-i for i in accuracy]

    print('Accuracy at conver = \n ', accuracy)
    print('Misclassification rate at conver = \n ', misclass)
    

    y_prob=treeFit.predict_proba(x_test)
    auc = metrics.roc_auc_score(y_test, y_prob[:,1])
    print("AUC: ", auc)

    for i in range(len(y_prob)):
        if y_prob[i][1]>0.2:
            y_prob[i][1]=1
        else:
            y_prob[i][1]=0
    print(y_prob)    
    test['predict_prob_1']=y_prob[:,1]
    boxplot = test.boxplot(column='predict_prob_1',by='quality_grp')
    
    

import statsmodels.api as stats
import scipy
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 200)



def SWEEPOperator(pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    
    A = numpy.copy(inputM)
    diagA = numpy.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:, k] = 0.0 * A[:, k]
            ANext[k, :] = ANext[:, k]
        A = ANext
    return (A, aliasParam, nonAliasParam)



def build_mnlogit (fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = numpy.transpose(fullX).dot(fullX)
    invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-7)

    # Build a multinomial logistic model
    X = fullX.iloc[:, list(nonAliasParam)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method = 'newton', maxiter = 1000, gtol = 1e-6, full_output = True, disp = True)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    # The number of free parameters
    nYCat = thisFit.J
    thisDF = len(nonAliasParam) * (nYCat - 1)

    # Return model statistics
    return (thisLLK, thisDF, thisParameter, thisFit)


def sample_wr (inData):
    n = len(inData)
    outData = numpy.empty((n,1))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData


def question_2():
    
    # part a

    # Forward Selection
    devianceTable = pd.DataFrame()
    # Model 0: y = Intercept
    y = train['quality_grp'].astype('category')
    # fins the categories of thsi categorival dtype
    y_category = y.cat.categories
    u = pd.DataFrame()
    u = y_train.isnull()
    designX = pd.DataFrame(u.where(u, 1)).rename(columns = {'quality_grp': "const"})
    LLK0, DF0, fullParams0, thisFit = build_mnlogit(designX, y)
    devianceTable = devianceTable.append([[0, 'Intercept', DF0, LLK0, None, None, None]])
    print(devianceTable)
    # find the pValues of all possible combinations and the one with smallest value determines Model
    allFeatures = ['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']
    catTarget = 'quality_grp'



    #print(allFeatures)
    nComb = len(allFeatures)
    
    #step 1.i try all features
    for i in allFeatures:
        designX = train[i]
        designX = stats.add_constant(designX, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)
        devianceTable = devianceTable.append([[1.1, 'Intercept + '+str(i),
                                       DF1, LLK1, testDev, testDF, testPValue]])
        print("\n")
        print("added:", i)
        print("\n")
        print(devianceTable)
    
    
    # Step 1: Intercept + 'Alcohol'
    designX = train['alcohol']
    designX = stats.add_constant(designX, prepend = True)
    LLK0, DF0, fullParams1, thisFit = build_mnlogit (designX, y)
    devianceTable = devianceTable.append([[1, 'Intercept + Alcohol', DF0, LLK0, None, None, None]])
    
    allFeatures.remove('alcohol')
    print("\n")
    print("final added: alcohol")
    print("\n")
    print(allFeatures)
    
    #step 2.i Intercept + 'Alcohol' + i
    for i in allFeatures:
        designX = train['alcohol'].to_frame()
        designX = designX.join(train[i])
        designX = stats.add_constant(designX, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)
        devianceTable = devianceTable.append([[2.1, 'Intercept + alcohol + '+str(i),
                                       DF1, LLK1, testDev, testDF, testPValue]])
        print("\n")
        print("added:", i)
        print("\n")
        print(devianceTable)
        
    # Step 2: Intercept + 'Alcohol' + 'free_sulfuf_dioxide'
    designX = train['alcohol'].to_frame().join(train['free_sulfur_dioxide'])
    designX = stats.add_constant(designX, prepend = True)
    LLK0, DF0, fullParams1, thisFit = build_mnlogit (designX, y)
    devianceTable = devianceTable.append([[2, 'Intercept + Alcohol+ free_sulfur_dioxide', DF0, LLK0, None, None, None]])
    
    allFeatures.remove('free_sulfur_dioxide')
    print("\n")
    print("final added: free_sulfur_dioxide")
    print("\n")
    print(allFeatures)
    
    #step 3.i Intercept + 'Alcohol' + 'free_sulfuf_dioxide' + i
    for i in allFeatures:
        designX = train['alcohol'].to_frame()
        designX = designX.join(train['free_sulfur_dioxide'])
        designX = designX.join(train[i])
        designX = stats.add_constant(designX, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)
        devianceTable = devianceTable.append([[3.1, 'Int + alc + fsd + '+str(i),
                                       DF1, LLK1, testDev, testDF, testPValue]])
        print("\n")
        print("added:", i)
        print("\n")
        print(devianceTable)

    # Step 3: Intercept + 'Alcohol' + 'free_sulfuf_dioxide' + sulphates
    designX = train['alcohol'].to_frame().join(train['free_sulfur_dioxide']).join(train['sulphates'])
    designX = stats.add_constant(designX, prepend = True)
    LLK0, DF0, fullParams1, thisFit = build_mnlogit (designX, y)
    devianceTable = devianceTable.append([[3, 'Intercept + Alcohol+ free_sulfur_dioxide', DF0, LLK0, None, None, None]])
    
    allFeatures.remove('sulphates')
    print("\n")
    print("final added: sulphates")
    print("\n")
    print(allFeatures)
    
    #step 4.i Intercept + 'Alcohol' + 'free_sulfuf_dioxide' + sulphates + i
    for i in allFeatures:
        designX = train['alcohol'].to_frame()
        designX = designX.join(train['free_sulfur_dioxide']).join(train['sulphates'])
        designX = designX.join(train[i])
        designX = stats.add_constant(designX, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)
        devianceTable = devianceTable.append([[4.1, 'Int + alc + fsd + sul + '+str(i),
                                       DF1, LLK1, testDev, testDF, testPValue]])
        print("\n")
        print("added:", i)
        print("\n")
        print(devianceTable)

    # Step 4: Intercept + 'Alcohol' + 'free_sulfuf_dioxide' + sulphates + citric_acid
    designX = train['alcohol'].to_frame().join(train['free_sulfur_dioxide']).join(train['sulphates']).join(train['citric_acid'])
    designX = stats.add_constant(designX, prepend = True)
    LLK0, DF0, fullParams1, thisFit = build_mnlogit (designX, y)
    devianceTable = devianceTable.append([[4, 'Int + Alc + fsd + citric_acid', DF0, LLK0, None, None, None]])
    
    allFeatures.remove('citric_acid')
    print("\n")
    print("final added: citric_acid")
    print("\n")
    print(allFeatures)
    
    #step 5.i Intercept + 'Alcohol' + 'free_sulfuf_dioxide' + sulphates + i
    for i in allFeatures:
        designX = train['alcohol'].to_frame()
        designX = designX.join(train['free_sulfur_dioxide']).join(train['sulphates']).join(train['citric_acid'])
        designX = designX.join(train[i])
        designX = stats.add_constant(designX, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)
        devianceTable = devianceTable.append([[5.1, 'Int + alc + fsd + sul + c_a +'+str(i),
                                       DF1, LLK1, testDev, testDF, testPValue]])
        print("\n")
        print("added:", i)
        print("\n")
        print(devianceTable)

    # Step 5: Intercept + 'Alcohol' + 'free_sulfuf_dioxide' + sulphates + citric_acid + residual_sugar
    designX = train['alcohol'].to_frame().join(train['free_sulfur_dioxide']).join(train['sulphates']).join(train['citric_acid']).join(train['residual_sugar'])
    designX = stats.add_constant(designX, prepend = True)
    LLK0, DF0, fullParams1, thisFit = build_mnlogit (designX, y)
    devianceTable = devianceTable.append([[5, 'Int + Alc + fsd + citric_acid + residual_sugar', DF0, LLK0, None, None, None]])
    
    print("\n")
    print("final added: residual_sugar")
    print("\n")

    # part b
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    y_res= clf.predict(x_test)
    auc = metrics.roc_auc_score(y_test, y_res)
    print("AUC: ", auc)

    # part c
    indices = [i for i in range(0,4547)]
    bs_i=[]
    for i in range(10000):
        bs_i.append(sample_wr(indices))
    for i in range(len(bs_i)):
        bs_i[i]=bs_i[i].flatten().tolist()
        bs_i[i]=[int(j)for j in bs_i[i]]

    bootstraps=[pd.DataFrame(columns=train.columns) for i in range(10000)]
    
    for i in range(len(bootstraps)):
        bootstraps[i]=train.iloc[bs_i[i]]
    aucs=[]
    for i in bootstraps:
        clf = LogisticRegression(random_state=0).fit(i.drop(['quality_grp'],axis=1), i['quality_grp'])
        y_res= clf.predict(x_test)
        auc = metrics.roc_auc_score(y_test, y_res)
        aucs.append(auc)
    plt.figure(2)
    plt.hist(aucs,bins=40)
    plt.show()
    print('95% Confidence Interval: {:.7f}, {:.7f}'.format(numpy.percentile(aucs, (2.5)), numpy.percentile(aucs, (97.5))))


question_1()
train=train.drop(['predCluster'],axis=1)
question_2()



