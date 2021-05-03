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
def sample_wr(inData):
    n = len(inData)
    outData = np.empty((n,6))
    for i in range(n):
        j = int(random.random() * n)
        outData[i] = inData[j]
    return outData

#---------------------------------------------

# Read Data -------------------------------
trainData = pandas.read_csv('WineQuality_Train.csv')
testData = pandas.read_csv('WineQuality_Test.csv')
# -------------------------------
nObs = trainData.shape[0]

x_train = trainData[INPUT_FEATURES]
y_train = trainData[TARGET_FEATURE]

x_test = trainData[INPUT_FEATURES]
y_test = trainData[TARGET_FEATURE]


# q1.a -------------------------------



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
# -------------------------------
