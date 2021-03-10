# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:29:21 2021

@author: pc
"""

# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the TREE library from SKLEARN
from sklearn import tree
import math
from itertools import combinations

# def findsubsets(data, size): 
#     return list(itertools.combinations(data, size))


def split(data2, split = 1, typ = 'EO', subset = None): 
    data = data2.copy()
    if typ == 'EO':
        data['LE_Split'] = (data.iloc[:,0] <= split)
    elif typ == 'EN':
        data['LE_Split'] = data.iloc[:, 0].apply(lambda x: True if x in subset else False)
    cross_table = pd.crosstab(index=data['LE_Split'], 
                              columns=data.iloc[:, 1], margins=True, 
                              dropna=True)
    n_rows = cross_table.shape[0]
    n_col = cross_table.shape[1]
    t_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_col):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_col - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
        t_entropy += row_entropy * cross_table.iloc[i_row, (n_col - 1)]
    t_entropy = t_entropy / cross_table.iloc[(n_rows - 1), (n_col - 1)]
    return cross_table, t_entropy

def FindMinEO(data, intervals): 
    min_entropy = 999999999.99999999999
    min_interval = None
    min_table = None
    for i in range(intervals[0], intervals[len(intervals) - 1]):
        cur_table, cur_entropy = split(data, i + 0.5, typ = 'EO')
        if cur_entropy < min_entropy:
            min_entropy = cur_entropy
            min_interval = i + 0.5
            min_table = cur_table
    return min_table, min_entropy, min_interval

def FindMinEN(data, sett):
    subset_map = {}
    for i in range(1, (int(len(sett) / 2)) + 1):
        subsets = combinations(sett, i)
        for ss in subsets:
            remaining = tuple()
            for ele in sett:
                if ele not in ss:
                    remaining += (ele,)
            if subset_map.get(remaining) == None:
                subset_map[ss] = remaining
    min_entropy = 99999999999.99999999999
    min_subset1 = min_subset2 = min_table = None
    for subsett in subset_map:
        table, entropy = split(data, typ = 'EN', subset = subsett)
        if entropy < min_entropy:
            min_entropy = entropy
            min_subset1 = subsett
            min_subset2 = subset_map.get(subsett)
            min_table = table
    return min_table, min_entropy, min_subset1, min_subset2




train_data = pd.read_csv('claim_history.csv')

n =train_data.shape[0]

# 2.a
p_com_train = train_data.groupby('CAR_USE').size()['Commercial'] / train_data.shape[0]
p_pri_train = train_data.groupby('CAR_USE').size()['Private'] / train_data.shape[0]
root_entropy = -((p_com_train * math.log2(p_com_train)) + (p_pri_train * math.log2(p_pri_train)))
print(f'Entropy of root node : {root_entropy}')

# 2.b
# train_data = train_data[['CAR_USE','CAR_TYPE', 'OCCUPATION', 'EDUCATION']].dropna()
# uniqueCarType = train_data['CAR_TYPE'].unique()

# subsets = findsubsets(uniqueCarType, 1)
# sub = pandas.DataFrame(columns=train_data.columns)
# for types in subsets[0]:
#     left = sub.append(train_data[train_data['CAR_TYPE'] == types])
    
# n_sub = sub.shape[0]

# # entropy = - ((n_sub/n)*math.log2() + (train_data.shape[0] -sub.shape[0])*math.log2((train_data.shape[0] -sub.shape[0]))) 
# EntropySplit(train_data, 'Minivan')

train_data['EDUCATION'] = train_data['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

education_split = FindMinEO(train_data[['EDUCATION', 'CAR_USE']],[0, 1, 2, 3, 4])
car_type_split = FindMinEN(train_data[['CAR_TYPE', 'CAR_USE']], 
                                       ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 
                                        'Sports Car', 'Van'])
occupation_split = FindMinEN(train_data[['OCCUPATION', 'CAR_USE']],
                             ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 
                              'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
print("Split Entropy of Education: {0}".format(education_split[1]))
print("Split Entropy of Car Type: {0}".format(car_type_split[1]))
print("Split Entropy of Occupation: {0}".format(occupation_split[1]))
print("Left Branch: {0}".format(occupation_split[2]))
print("Right Branch: {0}".format(occupation_split[3]))


train_left_split = train_data[train_data['OCCUPATION'].isin(occupation_split[2])]
left_educ_split = FindMinEO(train_left_split[['EDUCATION', 'CAR_USE']],
                            [0, 1, 2, 3, 4])
left_car_split = FindMinEN(train_left_split[['CAR_TYPE', 'CAR_USE']], 
                                       ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 
                                        'Sports Car', 'Van'])
left_oc_split = FindMinEN(train_left_split[['OCCUPATION', 'CAR_USE']],
                             ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 
                              'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])

print("Split Entropy of Education in the next layer (left): {0}".format(left_educ_split[1]))
print("Split Entropy of Car Type in the next layer (left): {0}".format(left_car_split[1]))
print("Split Entropy of Occupation in the next layer (left): {0}".format(left_oc_split[1]))
# print("Left Branch (left): {0}".format(left_educ_split[2]))
# print("Right Branch (left): {0}".format(left_educ_split[3]))


train_right_split = train_data[train_data['OCCUPATION'].isin(occupation_split[3])]
right_educ_split = FindMinEO(train_right_split[['EDUCATION', 'CAR_USE']],
                            [0, 1, 2, 3, 4])
right_car_split = FindMinEN(train_right_split[['CAR_TYPE', 'CAR_USE']], 
                                       ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 
                                        'Sports Car', 'Van'])
right_oc_split = FindMinEN(train_right_split[['OCCUPATION', 'CAR_USE']],
                             ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 
                              'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])

print("Split Entropy of Education in the next layer (right): {0}".format(right_educ_split[1]))
print("Split Entropy of Car Type in the next layer (right): {0}".format(right_car_split[1]))
print("Split Entropy of Occupation in the next layer (right): {0}".format(right_oc_split[1]))
print("Left Branch (right): {0}".format(right_car_split[2]))
print("Right Branch (right): {0}".format(right_car_split[3]))


leave1_data = train_left_split[train_left_split['EDUCATION'] <= left_educ_split[1]]
print("Leave 1")
print("{0} --> {1}".format(occupation_split[2], '(Below High School)'))
print("Total count: {0}".format(leave1_data.shape[0]))
print("Commercial: {0}".format(leave1_data.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(leave1_data.groupby('CAR_USE').size()['Private']))

leave2_data = train_left_split[train_left_split['EDUCATION'] > left_educ_split[1]]
print("Leave 2")
print("{0} --> {1}".format(occupation_split[2], '(High School, Bachelors, Masters, Doctors)'))
print("Total count: {0}".format(leave2_data.shape[0]))
print("Commercial: {0}".format(leave2_data.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(leave2_data.groupby('CAR_USE').size()['Private']))


leave3_data = train_right_split[train_right_split['CAR_TYPE'].isin(right_car_split[2])]
print("Leave 3")
print("{0} --> {1}".format(occupation_split[3], right_car_split[2]))
print("Total count: {0}".format(leave3_data.shape[0]))
print("Commercial: {0}".format(leave3_data.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(leave3_data.groupby('CAR_USE').size()['Private']))



leave4_data = train_right_split[train_right_split['CAR_TYPE'].isin(right_car_split[3])]
print("Leave 4")
print("{0} --> {1}".format(occupation_split[3], right_car_split[3]))
print("Total count: {0}".format(leave4_data.shape[0]))
print("Commercial: {0}".format(leave4_data.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(leave4_data.groupby('CAR_USE').size()['Private']))





