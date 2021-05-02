#!/usr/bin/env python
# coding: utf-8

# In[1]:


#question 1


# In[2]:


import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.metrics as metrics


# In[3]:


trainData = pandas.read_csv('C:\\Users\\khura\\Python\\ML5\\WineQuality_Train.csv', delimiter=',')
testData = pandas.read_csv('C:\\Users\\khura\\Python\\ML5\\WineQuality_Test.csv', delimiter=',')


# In[4]:


nObs = trainData.shape[0]


# In[5]:


x_train = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = trainData['quality_grp']
x_test = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = trainData['quality_grp']


# In[6]:


classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
print('Accuracy = ', accuracy)
print('Misclassification Rate Iteration 0 = ', 1-accuracy)


# In[7]:


w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(50)
ensemblePredProb = numpy.zeros((nObs, 2))

for iter in range(50):
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

misclass=1-accuracy
misclass


# In[8]:


x_test = testData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = testData['quality_grp']
nObs1 = testData.shape[0]


# In[9]:


y_score = treeFit.predict_proba(x_test)
AUC = metrics.roc_auc_score(y_test,y_score[:,1])
print('AUC = ', AUC)


# In[10]:


df1=pandas.DataFrame(y_test)
df2=pandas.DataFrame(treePredProb[:,1])
df3= pandas.concat([df1, df2], axis=1, join='inner')
df3 = df3.rename(columns={0:"Pred"})
df3.boxplot(column='Pred', by='quality_grp' ,figsize=(5,5))


# In[11]:


#question 2


# In[14]:


import scipy
import numpy as np
import pandas as pd
import statsmodels.api as stats
from itertools import combinations


train = pd.read_csv('WineQuality_Train.csv')
test = pd.read_csv('WineQuality_Test.csv')
x_train = train[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]
x_test = test[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]
y_train = train['quality_grp']
y_test = test['quality_grp']
nObs = train.shape[0]

# part a

def SWEEPOperator(pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    
    A = np.copy(inputM)
    diagA = np.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - np.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:, k] = 0.0 * A[:, k]
            ANext[k, :] = ANext[:, k]
        A = ANext
    return (A, aliasParam, nonAliasParam)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit(fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = np.transpose(fullX).dot(fullX)
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

chi_dict = {}

# Forward Selection

# Model 0: y = Intercept
y = train['quality_grp'].astype('category')
# fins the categories of thsi categorival dtype
y_category = y.cat.categories
u = pd.DataFrame()
u = y_train.isnull()
designX = pd.DataFrame(u.where(u, 1)).rename(columns = {'quality_grp': "const"})
LLK0, DF0, fullParams0, thisFit = build_mnlogit(designX, y)

# find the pValues of all possible combinations and the one with smallest value determines Model
allFeature = ['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']
catTarget = 'quality_grp'
allCombResult = pd.DataFrame()
allComb = []
for r in range(len(allFeature)+1):
   allComb = allComb + list(combinations(allFeature, r))
nComb = len(allComb)
Y = train[catTarget].astype('category') 
for r in range(1,nComb):
   modelTerm = list(allComb[r])
   trainData = train[modelTerm].dropna()
   trainData = stats.add_constant(trainData, prepend = True)
   LLK1, DF1, fullParams1, thisFit = build_mnlogit(trainData, y)
   testDev = 2.0 * (LLK1 - LLK0)
   testDF = DF1 - DF0
   testPValue = scipy.stats.chi2.sf(testDev, testDF)  
   chi_dict[allComb[r]] = testPValue
key_min = min(chi_dict.keys(), key=(lambda k: chi_dict[k]))
print('Model = Intercept +',' + '.join(''.join(t) for t in key_min))


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import sklearn.tree as tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logistic = LogisticRegression(random_state = 20210415).fit(x_train,y_train)
pred_prob = logistic.predict_proba(x_test)
AUC = metrics.roc_auc_score(y_test, pred_prob[:,1])       
print('AUC = ', AUC)


# In[16]:


train_sample = pd.concat([x_train,y_train],axis=1)
train_sample = train_sample.values

random.seed(20210415)
def sample_wr(inData):
    n = len(inData)
    outData = np.empty((n,6))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

#bootstrap_samples = np.zeros((10000,4547,6))
AUC_array = np.zeros(10000)
for i in range(10000):
    bootstrap = sample_wr(train_sample) 
    x_train = bootstrap[:,:5]
    y_train = bootstrap[:,-1]
    logistic = LogisticRegression(random_state = 20210415).fit(x_train,y_train)
    pred_prob = logistic.predict_proba(x_test)
    AUC_array[i] = metrics.roc_auc_score(y_test, pred_prob[:,1])


# In[17]:


print('95% Confidence Interval: {:.7f}, {:.7f}' .format(numpy.percentile (AUC_array, (2.5)), numpy.percentile (AUC_array,(97.5))))
plt.hist(AUC_array, bins=numpy.arange (min (AUC_array), max(AUC_array) +0.001, 0.001), align='mid',edgecolor='blue', color='red')
plt.grid(axis='both') 
plt.xlabel('AUC Value')
plt.ylabel('No of observations')
plt.show()


# In[ ]:




