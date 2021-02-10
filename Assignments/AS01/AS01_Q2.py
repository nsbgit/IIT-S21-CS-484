# -*- coding: utf-8 -*-
"""

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Splring 2021
"""

# Load the necessary libraries
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree


def DetectOutliers(lbl, dt):
    dt = dt.to_frame()
    outliersDF = pd.DataFrame()
    q1 = dt['x'].quantile(q=0.25)
    q3 = dt['x'].quantile(q=0.75)
    iqr = q3 - q1
    lowerWhisker = q1 - 1.5 * iqr
    upperWhisker = q3 + 1.5 * iqr
    dt_list = list(dt['x'])
    outliers = []
    for element in dt_list:
        if element < lowerWhisker or element > upperWhisker:
            outliers.append(element)
    # print(outliers)
    # print('\n\n\n')
    # t_d = (data['x'][data['group'] == 0]).to_frame()
    ol = dt.loc[(dt['x'] < lowerWhisker) | (dt['x'] > upperWhisker)]['x']
    # print(type(ol))
    ol = ol.to_frame()
    # print(type(ol))
    # result = result.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin', 6:'uBin', 7:'binFreq'})
    # ol = ol.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin', 6:'uBin', 7:'binFreq'})
    print("\nThere are {0} outliers in the {1}. They are:".format(len(ol), lbl))
    print(ol)
    

# **************************  Question 2  ******************************************
data = pd.read_csv('NormalSample.csv')


# **************************  Question 2.a  ******************************************
print("\n\n\nQ2.a)		(5 points) What is the five-number summary of x for each category of the group? What are the values of the 1.5 IQR whiskers for each category of the group?\n")
groups = data['group'].unique()
dataDescribe = pd.DataFrame()
dataSummary = pd.DataFrame()
whiskers = pd.DataFrame()
for group in groups:
    describe = data['x'][data['group'] == group].describe()
    dataDescribe = dataDescribe.append(
        {'Group':int(group), 
            'Describe':describe}
        , ignore_index=True)
    dataSummary = dataSummary.append(
        {'Group':int(group),
         'Minumum':data['x'][data['group'] == group].min(),
         'Q1':data['x'][data['group'] == group].quantile(q=0.25),
         'Median':data['x'][data['group'] == group].quantile(q=0.50),
         'Q3':data['x'][data['group'] == group].quantile(q=0.75),
         'Maximum':data['x'][data['group'] == group].max()},
        ignore_index=True)
    q1 = data['x'][data['group'] == group].quantile(q=0.25)
    q3 = data['x'][data['group'] == group].quantile(q=0.75)
    iqr = q3 - q1
    lowerWhisker = q1 - 1.5 * iqr
    upperWhisker = q3 + 1.5 * iqr
    whiskers = whiskers.append(
        {'Group':group,
         'IQR':iqr,
         'Lower Whisker':lowerWhisker,
         'Upper Whisker':upperWhisker},
        ignore_index=True)

# sortedResult = result.sort_values(by=['C(Delta)']).reset_index(drop=True)
dataDescribe = dataDescribe.sort_values(by=['Group']).reset_index(drop=True)
dataSummary = dataSummary.sort_values(by=['Group']).reset_index(drop=True)[['Group','Minumum','Q1','Median','Q3','Maximum']]
dataSummary.head()
print("\nThe below figure shows five-number summary of x for each category of the group:\n")
print(dataSummary)
print("\nThe values of the 1.5 IQR whiskers from each category of group are as follows:\n")
whiskers = whiskers.sort_values(by=['Group']).reset_index(drop=True)
print(whiskers)

# **************************  Question 2.b  ******************************************
boxplotData = [data['x']]
tickValue = ['','Overall']
for group in groups:
    dt = data['x'][data['group'] == group]
    boxplotData.append(dt)
    tickValue.append("Group {0}".format(group))
plt.boxplot(boxplotData, vert=0,patch_artist= True)
plt.title("Box Plot of X for each category of Group")
plt.yticks(np.arange(len(tickValue)), tuple(tickValue))
plt.xlabel("X")
plt.ylabel("Category")
plt.grid(axis="y")
plt.grid(axis="x")
plt.show()

#  Detect outliers
# outliersDF = pd.DataFrame()
# q1 = data['x'][data['group'] == 0].quantile(q=0.25)
# q3 = data['x'][data['group'] == 0].quantile(q=0.75)
# iqr = q3 - q1
# lowerWhisker = q1 - 1.5 * iqr
# upperWhisker = q3 + 1.5 * iqr
# dt_list = list(data['x'][data['group'] == 0])
# outliers = []
# for element in dt_list:
#     if element < lowerWhisker or element > upperWhisker:
#         outliers.append(element)
# print(outliers)
# print('\n\n\n')
# t_d = (data['x'][data['group'] == 0]).to_frame()
# ol = t_d.loc[(t_d['x'] < lowerWhisker) | (t_d['x'] > upperWhisker)]['x']
# print(ol)
DetectOutliers('Overall', data['x'])
for group in groups:
    DetectOutliers("Group {0}".format(group), data['x'][data['group'] == group])






# **************************  Question 2  ******************************************
# **************************  Question 2  ******************************************
# **************************  Question 2  ******************************************
# **************************  Question 2  ******************************************
# **************************  Question 2  ******************************************
# **************************  Question 2  ******************************************


