# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:46:49 2018

@author: suhas
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import sklearn.svm as svm


df = pd.read_csv('WineQuality.csv')


df = df.drop(columns=['quality', 'type'])
nTotal = len(df)
df.shape

groups = df.groupby('quality_grp')    
seper = [groups.get_group(x) for x in groups.groups]    

column = list(df) 
column.pop()


for i in range(len(column)):
    to_plot = pd.DataFrame([seper[1].iloc[:,i], seper[0].iloc[:,i]] , index=['1', '0'])
    to_plot.T.boxplot(vert=False)
    plt.title("Boxplot of " + str(column[i]) + " grouped by quality_grp") 
    plt.xlabel(str(column[i]))
    plt.ylabel('quality_grp')
    plt.show()  
    
l_s_p = pd.DataFrame(columns=['lab','statistic','p_value'])
y=0

for i in column: 
    grp1=df[df['quality_grp']==1][i]
    grp0=df[df['quality_grp']==0][i]
    t_test = stats.ttest_ind(grp1, grp0)
    print("\n",i,"\n",t_test)
    l_s_p.loc[y] = [i, t_test[0], t_test[1]]
    y = y + 1

l_s_p = l_s_p.sort_values('p_value', ascending=False)  


drp_qual_grp = df.drop(columns=['quality_grp'])
label = df['quality_grp']
    
drp_sulfur = drp_qual_grp.drop(columns=['free_sulfur_dioxide'])
drp_ph = drp_sulfur.drop(columns=['pH'])
drp_sulphates = drp_ph.drop(columns=['sulphates'])
drp_fixedacid = drp_sulphates.drop(columns=['fixed_acidity'])
drp_sulfdiox = drp_fixedacid.drop(columns=['total_sulfur_dioxide'])
drp_citric = drp_sulfdiox.drop(columns=['citric_acid'])
drp_sug = drp_citric.drop(columns=['residual_sugar'])

left = list(drp_sug)

print("Input attributes are retained such that the algorithm can converge for the first time\n",left)
svm_Model = svm.LinearSVC(random_state = 20181111, max_iter = 10000)
thisFit = svm_Model.fit(drp_sug, label)


print('Intercept:\n', thisFit.intercept_)
print('Weight Coefficients:\n', thisFit.coef_)

predictions = thisFit.predict(drp_sug)

print('Mean Accuracy = ', metrics.accuracy_score(label, predictions))
drp_sug['_PredictedClass_'] = predictions
svm_Mean = drp_sug.groupby('_PredictedClass_').mean()
print(svm_Mean)

dat = drp_sug.drop(columns=['_PredictedClass_'])

details = dat.describe()
mean_attr = np.array(details.iloc[1,:])
twentyfive = np.array(details.iloc[4,:])
seventyfive = np.array(details.iloc[6,:])   
print(mean_attr)
print(twentyfive)
print(seventyfive)

f=pd.DataFrame(mean_attr.reshape(1,4),columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
g=pd.DataFrame(twentyfive.reshape(1,4),columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
h=pd.DataFrame(seventyfive.reshape(1,4),columns=['volatile_acidity', 'chlorides', 'density', 'alcohol'])
predictionsformean = thisFit.predict(f)
print(predictionsformean)
predictionsfor25 = thisFit.predict(g)
print(predictionsfor25)
predictionsfor75 = thisFit.predict(h)
print(predictionsfor75)


