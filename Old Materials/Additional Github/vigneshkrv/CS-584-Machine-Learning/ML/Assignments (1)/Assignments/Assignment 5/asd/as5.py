# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:32:07 2018

@author: visma
"""


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


df = pd.read_csv('WineQuality.csv')


df = df.drop(columns=['quality', 'type'])
nTotal = len(df)
df.shape

gb = df.groupby('quality_grp')    
divide = [gb.get_group(x) for x in gb.groups]    

column = list(df) 
column.pop()


for i in range(len(column)):
    pt = pd.DataFrame([divide[1].iloc[:,i], divide[0].iloc[:,i]] , index=['1', '0'])
    pt.T.boxplot(vert=False)
    plt.title(str(i+1)+ " Boxplot of " + str(column[i]) + " Grouped by quality_grp") 
    plt.xlabel(str(column[i]))
    plt.ylabel('quality_grp')
    plt.show()  
    
out = pd.DataFrame(columns=['lab','statistic','p_value'])
y=0

for i in column: 
    group_1=df[df['quality_grp']==1][i]
    group_0=df[df['quality_grp']==0][i]
    ou = stats.ttest_ind(group_1, group_0)
    print("\n",i,"\n",ou)
    out.loc[y] = [i, ou[0], ou[1]]
    y = y + 1

out = out.sort_values('p_value', ascending=False)  
import sklearn.svm as svm

X = df.drop(columns=['quality_grp'])
label = df['quality_grp']
    
X1 = X.drop(columns=['free_sulfur_dioxide'])
X2 = X1.drop(columns=['pH'])
X3 = X2.drop(columns=['sulphates'])
X4 = X3.drop(columns=['fixed_acidity'])
X5 = X4.drop(columns=['total_sulfur_dioxide'])
X6 = X5.drop(columns=['citric_acid'])
X7 = X6.drop(columns=['residual_sugar'])

left = list(X7)

print("Input attributes are retained such that the algorithm can converge for the first time\n",left)
svm_Model = svm.LinearSVC(random_state = 20181111, max_iter = 10000)
thisFit = svm_Model.fit(X7, label)


print('Intercept:\n', thisFit.intercept_)
print('Weight Coefficients:\n', thisFit.coef_)

y_predictClass = thisFit.predict(X7)

print('Mean Accuracy = ', metrics.accuracy_score(label, y_predictClass))
X7['_PredictedClass_'] = y_predictClass
svm_Mean = X7.groupby('_PredictedClass_').mean()
print(svm_Mean)

dat = X7.drop(columns=['_PredictedClass_'])

details = dat.describe()
attributes_mean = np.array(details.iloc[1,:])
attributes_25 = np.array(details.iloc[4,:])
attributes_75 = np.array(details.iloc[6,:])   
print(attributes_mean)
print(attributes_25)
print(attributes_75)

f1=pd.DataFrame(attributes_mean.reshape(1,4),columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
f2=pd.DataFrame(attributes_25.reshape(1,4),columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
f3=pd.DataFrame(attributes_75.reshape(1,4),columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
y_pred_mean = thisFit.predict(f1)
print(y_pred_mean)
y_pred_25 = thisFit.predict(f2)
print(y_pred_25)
y_pred_75 = thisFit.predict(f3)
print(y_pred_75)


