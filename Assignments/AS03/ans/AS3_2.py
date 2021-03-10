# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:29:21 2021

@author: pc
"""

# Load the necessary libraries
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Load the TREE library from SKLEARN
# from sklearn import tree
import math
from itertools import combinations
import sys

# def findsubsets(data, size):
#     return list(itertools.combinations(data, size))

conMin = 999999999.99999999999


def doSplitting(data2, split=1, predictor_type='Ordinal', subset=None):
    data = data2.copy()
    if predictor_type == 'Ordinal':
        data['LE_Split'] = (data.iloc[:, 0] <= split)
    elif predictor_type == 'Nominal':
        data['LE_Split'] = data.iloc[:, 0].apply(lambda x: True if x in subset else False)
    cross_table = pd.crosstab(index=data['LE_Split'], columns=data.iloc[:, 1], margins=True, dropna=True)
    rows = cross_table.shape[0]
    cols = cross_table.shape[1]
    entropy = 0
    for e in range(rows - 1):
        tempEntropy = 0
        for col in range(cols):
            proportion = cross_table.iloc[e, col] / cross_table.iloc[e, (cols - 1)]
            if proportion > 0:
                tempEntropy = tempEntropy - (proportion * np.log2(proportion))
        entropy = entropy + (tempEntropy * cross_table.iloc[e, (cols - 1)])
    entropy = entropy / cross_table.iloc[(rows - 1), (cols - 1)]
    return cross_table, entropy


def findMinEntropyForOrdinal(data, intervalList):
    minEntropy = conMin
    minInterval = None
    minTable = None
    for i in range(intervalList[0], intervalList[len(intervalList) - 1]):
        tempTable, tempEntropy = doSplitting(data, i + 0.5, predictor_type='Ordinal')
        if tempEntropy < minEntropy:
            minEntropy = tempEntropy
            minInterval = i + 0.5
            minTable = tempTable
    return minTable, minEntropy, minInterval


def findMinEntropyForNominal(data, intervalSet):
    subSetMap = {}
    for i in range(1, (int(len(intervalSet) / 2)) + 1):
        subsets = combinations(intervalSet, i)
        for ss in subsets:
            remaining = tuple()
            for ele in intervalSet:
                if ele not in ss:
                    remaining += (ele,)
            if subSetMap.get(remaining) is None:
                subSetMap[ss] = remaining
    minEntropy = conMin
    tempSubsetL = None
    tempSubsetR = None
    tempTable = None
    for e in subSetMap:
        table, entropy = doSplitting(data, predictor_type='Nominal', subset=e)
        if entropy < minEntropy:
            minEntropy = entropy
            tempSubsetL = e
            tempSubsetR = subSetMap.get(e)
            tempTable = table
    return tempTable, minEntropy, tempSubsetL, tempSubsetR


claimData = pd.read_csv('claim_history.csv')

n = claimData.shape[0]

# 2.a
comercialProb = claimData.groupby('CAR_USE').size()['Commercial'] / claimData.shape[0]
privateProb = claimData.groupby('CAR_USE').size()['Private'] / claimData.shape[0]
rootNodeEntropy = -((comercialProb * math.log2(comercialProb)) + (privateProb * math.log2(privateProb)))
print('Entropy of root node : {0}'.format(rootNodeEntropy))

# 2.b
# claimData = claimData[['CAR_USE','CAR_TYPE', 'OCCUPATION', 'EDUCATION']].dropna()
# uniqueCarType = claimData['CAR_TYPE'].unique()

# subsets = findsubsets(uniqueCarType, 1)
# sub = pandas.DataFrame(columns=train_data.columns)
# for types in subsets[0]:
#     left = sub.append(train_data[train_data['CAR_TYPE'] == types])

# n_sub = sub.shape[0]

# # entropy = - ((n_sub/n)*math.log2() + (train_data.shape[0] -sub.shape[0])*math.log2((train_data.shape[0] -sub.shape[0])))
# EntropySplit(train_data, 'Minivan')

claimData['EDUCATION'] = claimData['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

splitEducation = findMinEntropyForOrdinal(claimData[['EDUCATION', 'CAR_USE']], [0, 1, 2, 3, 4])
splitCarType = findMinEntropyForNominal(claimData[['CAR_TYPE', 'CAR_USE']],
                                        ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
splitOccupation = findMinEntropyForNominal(claimData[['OCCUPATION', 'CAR_USE']],
                                           ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager',
                                            'Professional', 'Student', 'Unknown'])
print("Split Entropy of Education: {0}".format(splitEducation[1]))
print("Split Entropy of Car Type: {0}".format(splitCarType[1]))
print("Split Entropy of Occupation: {0}".format(splitOccupation[1]))
print("Left Branch: {0}".format(splitOccupation[2]))
print("Right Branch: {0}".format(splitOccupation[3]))

splitLeft = claimData[claimData['OCCUPATION'].isin(splitOccupation[2])]
splitLeftEducation = findMinEntropyForOrdinal(splitLeft[['EDUCATION', 'CAR_USE']], [0, 1, 2, 3, 4])
splitLeftCar = findMinEntropyForNominal(splitLeft[['CAR_TYPE', 'CAR_USE']],
                                        ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
splitLeftOccupation = findMinEntropyForNominal(splitLeft[['OCCUPATION', 'CAR_USE']],
                                               ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager',
                                                'Professional', 'Student', 'Unknown'])

print("Split Entropy of Education in the next layer (left): {0}".format(splitLeftEducation[1]))
print("Split Entropy of Car Type in the next layer (left): {0}".format(splitLeftCar[1]))
print("Split Entropy of Occupation in the next layer (left): {0}".format(splitLeftOccupation[1]))

splitRight = claimData[claimData['OCCUPATION'].isin(splitOccupation[3])]
splitRightEducation = findMinEntropyForOrdinal(splitRight[['EDUCATION', 'CAR_USE']],
                                               [0, 1, 2, 3, 4])
splitRightCar = findMinEntropyForNominal(splitRight[['CAR_TYPE', 'CAR_USE']],
                                         ['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
splitRightOccupation = findMinEntropyForNominal(splitRight[['OCCUPATION', 'CAR_USE']],
                                                ['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager',
                                                 'Professional', 'Student', 'Unknown'])

print("Split Entropy of Education in the next layer (right): {0}".format(splitRightEducation[1]))
print("Split Entropy of Car Type in the next layer (right): {0}".format(splitRightCar[1]))
print("Split Entropy of Occupation in the next layer (right): {0}".format(splitRightOccupation[1]))
print("Left Branch (right): {0}".format(splitRightCar[2]))
print("Right Branch (right): {0}".format(splitRightCar[3]))

split1 = splitLeft[splitLeft['EDUCATION'] <= splitLeftEducation[1]]
print("Leaf 1")
print("{0} --> {1}".format(splitOccupation[2], '(Below High School)'))
print("Total count: {0}".format(split1.shape[0]))
print("Commercial: {0}".format(split1.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(split1.groupby('CAR_USE').size()['Private']))

split2 = splitLeft[splitLeft['EDUCATION'] > splitLeftEducation[1]]
print("Leaf 2")
print("{0} --> {1}".format(splitOccupation[2], '(High School, Bachelors, Masters, Doctors)'))
print("Total count: {0}".format(split2.shape[0]))
print("Commercial: {0}".format(split2.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(split2.groupby('CAR_USE').size()['Private']))

split3 = splitRight[splitRight['CAR_TYPE'].isin(splitRightCar[2])]
print("Leaf 3")
print("{0} --> {1}".format(splitOccupation[3], splitRightCar[2]))
print("Total count: {0}".format(split3.shape[0]))
print("Commercial: {0}".format(split3.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(split3.groupby('CAR_USE').size()['Private']))

split4 = splitRight[splitRight['CAR_TYPE'].isin(splitRightCar[3])]
print("Leaf 4")
print("{0} --> {1}".format(splitOccupation[3], splitRightCar[3]))
print("Total count: {0}".format(split4.shape[0]))
print("Commercial: {0}".format(split4.groupby('CAR_USE').size()['Commercial']))
print("Private: {0}".format(split4.groupby('CAR_USE').size()['Private']))
