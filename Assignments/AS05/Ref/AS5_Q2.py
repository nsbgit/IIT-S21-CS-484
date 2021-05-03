# -*- coding: utf-8 -*-
"""
Created on Sun May  1 21:54:43 2021

@author: Sukanta
"""

# Importig Libraries
import scipy
import statsmodels.api as stats
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression

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
BOOTSTRAP_SAMPLE = 10
HIST_BIN_WIDTH = 0.001


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
        Akk = A[k, k]
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



def sample_wr(inData):
    n = len(inData)
    outData = np.empty((n, 6))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData


def build_mnlogit(fullX, y):
    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = np.transpose(fullX).dot(fullX)
    invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim=nFullParam, inputM=XtX, tol=1e-7)

    # Build a multinomial logistic model
    X = fullX.iloc[:, list(nonAliasParam)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', maxiter=1000, gtol=1e-6, full_output=True, disp=True)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    # The number of free parameters
    nYCat = thisFit.J
    thisDF = len(nonAliasParam) * (nYCat - 1)

    # Return model statistics
    return (thisLLK, thisDF, thisParameter, thisFit)


# ---------------------------------------------

# Read Data -------------------------------
train_data = pd.read_csv('WineQuality_Train.csv')
test_data = pd.read_csv('WineQuality_Test.csv')
# -------------------------------
nObs = train_data.shape[0]
dct = {}

x_train = train_data[INPUT_FEATURES]
y_train = train_data[TARGET_FEATURE]

x_test = train_data[INPUT_FEATURES]
y_test = train_data[TARGET_FEATURE]

# q1.a -------------------------------

y = train_data['quality_grp'].astype('category')
y_category = y.cat.categories
df = pd.DataFrame()
df = y_train.isnull()
x_deg = pd.DataFrame(df.where(df, 1)).rename(columns={TARGET_FEATURE: "const"})
LLK0, DF0, FP0, thisFit = build_mnlogit(x_deg, y)

df_all_rslt = pd.DataFrame()
all = []

for r in range(len(INPUT_FEATURES) + 1):
    all = all + list(combinations(INPUT_FEATURES, r))

len_all = len(all)
Y = train_data[TARGET_FEATURE].astype('category')

for r in range(1, len_all):
    model = list(all[r])
    train_data_2 = train_data[model].dropna()
    train_data_2 = stats.add_constant(train_data_2, prepend=True)
    LLK1, DF1, FP1, thisFit = build_mnlogit(train_data_2, y)
    DEV = 2.0 * (LLK1 - LLK0)
    DF = DF1 - DF0
    PV = scipy.stats.chi2.sf(DEV, DF)
    dct[all[r]] = PV
min_key = min(dct.keys(), key=(lambda k: dct[k]))
print('Model = Intercept +', ' + '.join(''.join(t) for t in min_key))

# q2.b -------------------------------


logistic = LogisticRegression(random_state=INIT_RNDM_SEED).fit(x_train, y_train)
predicted_probability = logistic.predict_proba(x_test)
auc = metrics.roc_auc_score(y_test, predicted_probability[:, 1])
print(f'The Area Under Curve metric on the Testing data is {auc}')

# q2.c -------------------------------


train_sample = pd.concat([x_train, y_train], axis=1)
train_sample = train_sample.values

arr_auc = np.zeros(BOOTSTRAP_SAMPLE)

for i in range(BOOTSTRAP_SAMPLE):
    bootstrap = sample_wr(train_sample)
    x_train = bootstrap[:, :5]
    y_train = bootstrap[:, -1]
    logistic = LogisticRegression(random_state=INIT_RNDM_SEED).fit(x_train, y_train)
    predicted_probability = logistic.predict_proba(x_test)
    arr_auc[i] = metrics.roc_auc_score(y_test, predicted_probability[:, 1])

HIST_BINS = np.arange(min(arr_auc), max(arr_auc) + 0.001, 0.001)
plt.hist(
    arr_auc
    , bins=HIST_BINS
    , align='mid')
plt.grid(axis='both')
plt.xlabel('AUC Value')
plt.ylabel('No of observations')
plt.show()

# q2.d -------------------------------
prct_2_5 = np.percentile(arr_auc, (2.5))
prct_97_5 = np.percentile(arr_auc, (97.5))
confidence_interval_95 = '95% Confidence Interval is {:.7f}, {:.7f}'.format(prct_2_5, prct_97_5)
print(f'2.5th percentile is {prct_2_5} and 97.5th percentile is {prct_97_5}')