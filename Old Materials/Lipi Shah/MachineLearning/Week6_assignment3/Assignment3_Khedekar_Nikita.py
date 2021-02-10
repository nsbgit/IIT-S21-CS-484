#!/usr/bin/env python
# coding: utf-8

# In[50]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import sys
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[51]:


from sklearn.model_selection import train_test_split
df1 = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week6_assignment3\\claim_history.csv',delimiter=',')
predictor= df1[["CAR_TYPE","OCCUPATION","EDUCATION"]]
predictor_train,predictor_test,df1_train, df1_test = train_test_split(predictor,df1["CAR_USE"],test_size = 0.3, random_state = 27513,stratify = df1['CAR_USE'])


# In[52]:


len(df1_test)


# a)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Training partition?
# 

# In[53]:


crossTable1 = pd.crosstab(index = df1_train, columns = ["Counts"], margins = True, dropna=True)
crossTable1['Proportions']= 100*(crossTable1['Counts']/len(df1_train))
crossTable1= crossTable1.drop(columns = ['All'])
crossTable1





# b)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Test partition?

# In[54]:


crossTable1 = pd.crosstab(index = df1_test, columns = ["Counts"], margins = True, dropna=True)
crossTable1['Proportions']= 100*(crossTable1['Counts']/len(df1_train))
crossTable1= crossTable1.drop(columns = ['All'])
crossTable1


# c)	(5 points). What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?

# In[55]:


count=0
probability_training= len(predictor_train)/len(df1["CAR_USE"])
for i in df1["CAR_USE"]:
    if i!="Private":
        count=count+1
dependent_prob1=(probability_training*count/len(df1["CAR_USE"]))/(count/len(df1["CAR_USE"]))  

print("The probability that an observation is in the Training partition given that CAR_USE = Commercial is",dependent_prob1)



# d)	(5 points). What is the probability that an observation is in the Test partition given that CAR_USE = Private?

# In[56]:


c=0
probability_testing= len(predictor_test)/len(df1["CAR_USE"])
for i in df1["CAR_USE"]:
    if i!="Commercial":
        c=c+1
dependent_prob2=(probability_testing*c/len(df1["CAR_USE"]))/(c/len(df1["CAR_USE"]))  

print("The probability that an observation is in the Testing partition given that CAR_USE = Private is",dependent_prob2)


# adding new column Lable which contain CAR_USE data to dataframe predictor_train

# In[57]:


predictor_train["Labels"] = df1_train


# In[58]:


predictor_train.head()


# 2a)	(5 points). What is the entropy value of the root node?

# In[59]:


#root node entropy
count_Commercial=0
count_Private=0
for i in predictor_train['Labels']:
    if i=="Commercial":
        count_Commercial=count_Commercial+1
        Commercial_cars=count_Commercial
    else:
        count_Private=count_Private+1
        Private_cars=count_Private
        

prob_commercial_cars=Commercial_cars/len(predictor_train['Labels'])
prob_Private_car=Private_cars/len(predictor_train['Labels'])

root_entropy=-((prob_commercial_cars * np.log2(prob_commercial_cars) + prob_Private_car * np.log2(prob_Private_car)))
print("root node entropy:",root_entropy)


# In[60]:


prob_commercial_cars


# In[61]:


# Define a function to visualize the percent of a particular target category by an interval predictor
def EntropyIntervalSplit (
   inData,          # input data frame (predictor in column 0 and target in column 1)
   split):          # split value

    dataTable = inData
    dataTable['LE_Split'] = False
    for k in dataTable.index:
        if dataTable.iloc[:,0][k] in split:
            dataTable['LE_Split'][k] = True

    crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True,dropna = True)   
    
    nRows = crossTable.shape[0]
    nColumns = crossTable.shape[1]
    tableEntropy = 0
    
    for iRow in range(nRows-1):
        rowEntropy = 0
        for iColumn in range(nColumns):
            proportion = crossTable.iloc[iRow,iColumn] / crossTable.iloc[iRow,(nColumns-1)]
            if (proportion > 0):
                rowEntropy -= proportion * np.log2(proportion)
        tableEntropy += rowEntropy *  crossTable.iloc[iRow,(nColumns-1)]
    tableEntropy = tableEntropy /  crossTable.iloc[(nRows-1),(nColumns-1)]
  
    return(tableEntropy)


# In[62]:


def min_entropy(df,variable,combinations):
    inData1 = df[[variable,"Labels"]]
    entropies = []
    for i in combinations:
        EV = EntropyIntervalSplit(inData1, list(i))
        entropies.append((EV,i))
    return min(entropies)


# In[63]:


from sklearn import tree
from itertools import combinations
Car_type=df1.CAR_TYPE.unique()

car_type = []
for k in range(1,int((len(Car_type)/2)+1)):
    x = list(combinations(Car_type, k))
    if k == 3:
        x = x[:10]
    car_type.extend(x)
len(car_type)


# In[64]:


Occupation=df1.OCCUPATION.unique()
occupation = []
for k in range(1,(int((len(Occupation)/2))+1)):
    x = list(combinations(Occupation, k))
    occupation.extend(x)
len(occupation)


# In[65]:


entropy_car_type = min_entropy(predictor_train,"CAR_TYPE",car_type)
entropy_car_type


# In[66]:


entropy_occupation = min_entropy(predictor_train,"OCCUPATION",occupation)
entropy_occupation


# In[67]:


education = [('Below High School',),('Below High School','High School',),('Below High School','High School','Bachelors',),('Below High School','High School','Bachelors','Masters',),('Below High School','High School','Bachelors','Masters','Doctors',)]


# In[68]:


entropy_edu = min_entropy(predictor_train,"EDUCATION",education)
entropy_edu


# In[69]:


min_entropy_finder=min(entropy_edu[0],entropy_occupation[0],entropy_car_type[0])


# b)	(5 points). What is the split criterion (i.e., predictor name and values in the two branches) of the first layer?

# In[70]:


print(min_entropy_finder)


# e)	(15 points). Describe all your leaves.  Please include the decision rules and the counts of the target values.

# In[71]:


left_leaf_l1= predictor_train[(predictor_train["OCCUPATION"] == "Blue Collar")|(predictor_train["OCCUPATION"] == "Unknown")|(predictor_train["OCCUPATION"] == "Student")]
right_leaf_l1= predictor_train[(predictor_train["OCCUPATION"] != "Blue Collar")&(predictor_train["OCCUPATION"] != "Unknown")&(predictor_train["OCCUPATION"] != "Student")]


# In[72]:


left_leaf_l1_eduentropy=min_entropy(left_leaf_l1,"EDUCATION",education)
left_leaf_l1_eduentropy


# In[73]:


left_leaf_l1_CarTentropy=min_entropy(left_leaf_l1,"CAR_TYPE",car_type)
left_leaf_l1_CarTentropy


# In[74]:



occupationLL1 = []
for k in range(1,(int((len(list(entropy_occupation[1]))/2))+1)):
    x = list(combinations(list(entropy_occupation[1]), k))
    occupationLL1.extend(x)
occupationLL1


# In[75]:


occupationLL1_entropy=min_entropy(left_leaf_l1,"OCCUPATION",occupationLL1)
occupationLL1_entropy


# In[76]:


right_leaf_l1_eduentropy=min_entropy(right_leaf_l1,"EDUCATION",education)
right_leaf_l1_eduentropy


# In[77]:


right_leaf_l1_CarTentropy=min_entropy(right_leaf_l1,"CAR_TYPE",car_type)
right_leaf_l1_CarTentropy


# In[78]:


occupationRL1 = []
a=["Professional","Manager","Clerical","Doctor","Lawyer","Home Maker"]
for k in range(1,(int((len(a))/2)+1)):
    x = list(combinations(a, k))
    if k == 3:
        x = x[:10]
    occupationRL1.extend(x)
occupationRL1


# In[79]:


occupationRL1_entropy=min_entropy(right_leaf_l1,"OCCUPATION",occupationRL1)
occupationRL1_entropy


# 2nd split left

# In[80]:


left_leaf_ll2= left_leaf_l1[(left_leaf_l1["EDUCATION"] == "Below High School")]
left_leaf_rl2= left_leaf_l1[(left_leaf_l1["EDUCATION"] != "Below High School")]


# In[81]:


right_leaf_ll2=right_leaf_l1[(right_leaf_l1["CAR_TYPE"] == "Minivan")|(right_leaf_l1["CAR_TYPE"] == "SUV")|(right_leaf_l1["CAR_TYPE"] == "Sports Car")]
right_leaf_rl2=right_leaf_l1[(right_leaf_l1["CAR_TYPE"] != "Minivan")&(right_leaf_l1["CAR_TYPE"] != "SUV")&(right_leaf_l1["CAR_TYPE"] != "Sports Car")]


# probability of an event=Commercial for leftmost leaf at level 2:

# In[101]:


a=0
for i in (left_leaf_ll2['Labels']):
    if i=="Commercial":
        a=a+1
    prob_commercial_cars_leftll2=a/len(left_leaf_ll2['Labels'])
print("Count of commercial :",a,"Count of Private :",(len(left_leaf_ll2)-a),"and Predicted _probability ",prob_commercial_cars_leftll2)


# probability of an event=Commercial for 2nd leaf at level 2:

# In[102]:


a1=0
for i in (left_leaf_rl2['Labels']):
    if i=="Commercial":
        a1=a1+1
    prob_commercial_cars_leftrl2=a1/len(left_leaf_rl2['Labels'])
print("Count of commercial :",a1,"Count of Private :",(len(left_leaf_rl2)-a1),"and Predicted _probability ",prob_commercial_cars_leftrl2)


# probability of an event=Commercial for 3rd leaf at level 2:

# In[103]:


a2=0
for i in (right_leaf_ll2['Labels']):
    if i=="Commercial":
        a2=a2+1
    prob_commercial_cars_rightll2=a2/len(right_leaf_ll2['Labels'])
print("Count of commercial :",a2,"Count of Private :",(len(right_leaf_ll2)-a2),"and Predicted _probability ",prob_commercial_cars_rightll2)


# probability of an event=Commercial for rightmost leaf at level 2:

# In[104]:


a3=0
for i in (right_leaf_rl2['Labels']):
    if i=="Commercial":
        a3=a3+1
    prob_commercial_cars_rightrl2=a3/len(right_leaf_rl2['Labels'])
prob_commercial_cars_rightrl2
print("Count of commercial :",a3,"Count of Private :",(len(right_leaf_rl2)-a3),"and Predicted _probability ",prob_commercial_cars_rightrl2)


# DECISION TREE RULES

# In[114]:


predicted_probability=[]
occ1 = ["Professional","Manager","Clerical","Doctor","Lawyer","Home Maker",]
edu1 = ["Below High School",]
cartype1 = ["Van","Panel Truck","Pickup",]
for k in predictor_test.index:
    if predictor_test.iloc[:,1][k] not in occ1:
            if predictor_test.iloc[:,2][k] in edu1:
                predicted_probability.append(0.25)
            else:
                predicted_probability.append(0.85)
    else:
            if predictor_test.iloc[:,0][k] in cartype1:
                predicted_probability.append(0.54)
            else:
                predicted_probability.append(0.01) 


# Labeling leaves as Commercial/Private(if pred_prob < threshold of prob of commercial--> Private and vice a versa)

# In[116]:


predictions = []
for i in range(0,len(df1_test)):
    if predicted_probability[i] < prob_commercial_cars :
        predictions.append("Private")
    else:
        predictions.append("Commercial")


# Question 3
# Use the proportion of target Event value in the training partition as the threshold, what is the Misclassification Rate in the Test partition?

# In[107]:


from sklearn.metrics import accuracy_score
print("Missclassification Rate",1-accuracy_score(df1_test,predictions))


# What is the Root Average Squared Error in the Test partition?
# 
# Root Average Squared Error

# In[108]:


RASE = 0.0
for i in range (0,len(df1_test)):
    if df1_test.iloc[i] == "Commercial":
        RASE += (1-predicted_probability[i])**2
    else:
        RASE += (predicted_probability[i])**2
RASE = math.sqrt(RASE/len(df1_test))
RASE


# In[117]:


AUC = metrics.roc_auc_score(true_values, predicted_probability)
AUC


# In[118]:


OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(df1_test, predicted_probability, pos_label = 'Commercial')


# In[119]:


OneMinusSpecificity = np.append([0], OneMinusSpecificity)
Sensitivity = np.append([0], Sensitivity)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])


# In[135]:


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'yellow', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title("ROC")
plt.plot([0, 1], [0, 1], color = 'green', linestyle = ':',linewidth = 2)
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()


# In[ ]:




