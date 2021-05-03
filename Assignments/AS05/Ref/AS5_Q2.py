# -*- coding: utf-8 -*-
"""
Created on Sun May  1 21:54:43 2021

@author: Sukanta
"""

# Importig Libraries
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.metrics as metrics
import scipy
import numpy as np
import pandas as pd
import statsmodels.api as stats
from itertools import combinations
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

# -------------------------------

# global variables
SPLITTING_CRITERION = 'entropy'
MAXIMUM_TREE_DEPTH = 5
INIT_RNDM_SEED = 20210415
MAX_BOOSTING_ITR = 50
INTERRUPT_ACCURACY = 0.9999999
INPUT_FEATURES = ['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']
TARGET_FEATURE = 'quality_grp'
random.seed(INIT_RNDM_SEED)
# -------------------------------

# Functions
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

#---------------------------------------------

# Read Data -------------------------------
train = pandas.read_csv('WineQuality_Train.csv')
test = pandas.read_csv('WineQuality_Test.csv')
# -------------------------------
nObs = train.shape[0]
chi_dict = {}

x_train = train[INPUT_FEATURES]
y_train = train[TARGET_FEATURE]

x_test = train[INPUT_FEATURES]
y_test = train[TARGET_FEATURE]


# q1.a -------------------------------

y = train['quality_grp'].astype('category')
# finis the categories of thsi categorival dtype
y_category = y.cat.categories
u = pd.DataFrame()
u = y_train.isnull()
designX = pd.DataFrame(u.where(u, 1)).rename(columns = {'quality_grp': "const"})
LLK0, DF0, fullParams0, thisFit = build_mnlogit(designX, y)

allFeature = INPUT_FEATURES#['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']
catTarget = TARGET_FEATURE#'quality_grp'
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

# q2.b -------------------------------


logistic = LogisticRegression(random_state = INIT_RNDM_SEED).fit(x_train,y_train)
pred_prob = logistic.predict_proba(x_test)
AUC = metrics.roc_auc_score(y_test, pred_prob[:,1])       
print(f'The Area Under Curve metric on the Testing data is {AUC}')


# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
# -------------------------------
