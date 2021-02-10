# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:46:52 2020

@author: Lipi
"""
## Note:- I have taken reference from the professor sample code for some part of assignment.
########################### Import Statements #################################

import numpy
import pandas as pd
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import combinations



#import csv file claim_history.csv

claim_history = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week6_assignment3\\claim_history.csv',delimiter=',')
claim_history.shape
claim_history.isna().sum()

##################Question 1 A ################################################
# split into test and train dataset Using stratified simple random
print('################## Answer 1 A ##################################################')
predictor_variables = claim_history.iloc[:, [11,12,17]].values
target_variable = claim_history.iloc[:, 14].values

#split data into 75% train and 25% test dataset
X_train, X_test, y_train, y_test = train_test_split(predictor_variables, target_variable, test_size = 0.25, random_state = 60616, stratify = claim_history['CAR_USE'])

index = ['Row'+str(i) for i in range(1, len(y_train)+1)]
df_y_train = pd.DataFrame(y_train, index = index)

train_ans =pd.concat([df_y_train.groupby([0]).size(), df_y_train.groupby([0]).size() / y_train.shape[0]], axis=1)
train_ans.columns = ['Count', 'Proportion']
print(train_ans)

################### Question 1 B #################################################
print('################## Answer 1 B ##################################################')
index = ['Row'+str(i) for i in range(1, len(y_test)+1)]
df_y_test = pd.DataFrame(y_test, index = index)

test_ans =pd.concat([df_y_test.groupby([0]).size(), df_y_test.groupby([0]).size() / y_test.shape[0]], axis=1)
test_ans.columns = ['Count', 'Proportion']
print(test_ans)

################## Question 1 C ####################################################
print('################## Answer 1 C ##################################################')
prob_train = 0.75
prob_test = 0.25

prob_train_commercial = (train_ans['Proportion'][0] * prob_train)/((train_ans['Proportion'][0] * prob_train) + (test_ans['Proportion'][0] * prob_test))
print("The probability that an observation is in the Training partition given that CAR_USE = Commercial is :",prob_train_commercial)


################## Question 1 D ####################################################
print('################## Answer 1 D ##################################################')
prob_train = 0.75
prob_test = 0.25

prob_test_Private = (test_ans['Proportion'][1] * prob_test)/((train_ans['Proportion'][1] * prob_train) + (test_ans['Proportion'][1] * prob_test))
print("The probability that an observation is in the Test partition given that CAR_USE = Private is :",prob_test_Private)


############### Question 2 A ###########################################################3
print('################## Answer 2 A ##################################################') 

prob_commercial_train = (df_y_train.groupby([0]).size()[0]) / len(df_y_train)
prob_private_train =(df_y_train.groupby([0]).size()[1]) / len(df_y_train)

root_entropy = -((prob_commercial_train * math.log2(prob_commercial_train)) + (prob_private_train* math.log2(prob_private_train)))
print('Entropy of Root =', root_entropy)

#print('################## Answer 2 B ##################') 
  
#######################################################################
def minimualNomEnt(inData, set):
    subsetMap = {}
    for i in range(1, (int(len(set) / 2)) + 1):
        subsets = combinations(set, i)
        for ss in subsets:
            remaining = tuple()
            for ele in set:
                if ele not in ss:
                    remaining += (ele,)
            if subsetMap.get(remaining) == None:
                subsetMap[ss] = remaining

    minEntropy = sys.float_info.max
    minSubset1 = None
    minSubset2 = None
    minTable = None

    for subset in subsetMap:
        retTable, retEntropy = EntNomSplit(inData=inData, subset=subset)
        if retEntropy < minEntropy:
            minEntropy = retEntropy
            minSubset1 = subset
            minSubset2 = subsetMap.get(subset)
            minTable = retTable

    return minTable, minEntropy, minSubset1, minSubset2

#############################################################################
def EntNomSplit(inData, subset):
    dataTable = inData
    dataTable['LE_Split'] = dataTable.iloc[:, 0].apply(lambda x: True if x in subset else False)

    crossTable = pd.crosstab(index=dataTable['LE_Split'], columns=dataTable.iloc[:, 1], margins=True, dropna=True)
  
    n_rows = crossTable.shape[0]
    n_columns = crossTable.shape[1]

    tableEntropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = crossTable.iloc[i_row, i_column] / crossTable.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * numpy.log2(proportion)
        
        tableEntropy += row_entropy * crossTable.iloc[i_row, (n_columns - 1)]
    tableEntropy = tableEntropy / crossTable.iloc[(n_rows - 1), (n_columns - 1)]

    return crossTable, tableEntropy

##############################################################################3
def minimumOrdEnt(inData, setIntervals):
    minEntropy = sys.float_info.max
    minInterval = None
    minTable = None

    for i in range(setIntervals[0], setIntervals[len(setIntervals) - 1]):
        retTable, retEntropy = EntOrdSplit(inData=inData, split=i + 0.5)
        if retEntropy < minEntropy:
            minEntropy = retEntropy
            minInterval = i + 0.5
            minTable = retTable

    return minTable, minEntropy, minInterval


###############################################################################
def EntOrdSplit(inData, split):
    dataTable = inData
    dataTable['LE_Split'] = (dataTable.iloc[:, 0] <= split)

    crossTable = pd.crosstab(index=dataTable['LE_Split'], columns=dataTable.iloc[:, 1], margins=True, dropna=True)

    n_rows = crossTable.shape[0]
    n_columns = crossTable.shape[1]

    tableEntropy = 0 
    
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = crossTable.iloc[i_row, i_column] / crossTable.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * numpy.log2(proportion)

        tableEntropy += row_entropy * crossTable.iloc[i_row, (n_columns - 1)]
    tableEntropy = tableEntropy / crossTable.iloc[(n_rows - 1), (n_columns - 1)]

    return crossTable, tableEntropy


#create data frame with required columns
claim_history2 = claim_history[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()

#mapping Education category with [0,1,2,3,4] accordingly
claim_history2['EDUCATION'] = claim_history2['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
#print(claim_history2)

#divide test and train dataset as per required criteria 25:75
claim_history2_train, claim_history2_test = train_test_split(claim_history2, test_size = 0.25, random_state=60616, stratify = claim_history['CAR_USE'])


# for layer 1 split
crossTable_edu, tableEntropy_edu, interval_edu = minimumOrdEnt(inData=claim_history2_train[['EDUCATION', 'CAR_USE']],
                                                                         setIntervals=[0, 1, 2, 3, 4])


crossTable_car_type, tableEntropy_car_type, subset1_car_type, subset2_car_type = minimualNomEnt(
    inData=claim_history2_train[['CAR_TYPE', 'CAR_USE']], set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])

crossTable_occu, tableEntropy_occu, subset1_occu, subset2_occu = minimualNomEnt(
    inData=claim_history2_train[['OCCUPATION', 'CAR_USE']],
    set=['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])




#find entropy for all 3 predictor attribute and their minimal split
print('\n ################## Answer 2 B ##################################################') 
print('Entropy for atribute Education =', tableEntropy_edu)
print('Entropy for atribute Car_Type =', tableEntropy_car_type)
print('Entropy for atribute Ocuupation =', tableEntropy_occu)
print("Taken the minimal entropy as attribute that is Occupation for first layer spliting tree")
print("Layer 1 left split = ", subset1_occu)
print("Layer 1 right split = ", subset2_occu)


print('################## Answer 2 C ##########################################') 
print("The entropy of the split of the first layer is =",tableEntropy_occu)



print('################## Answer 2 D ##########################################') 
print("Maximum given depth = 2")
print("Number of leaves nodes = 2 ^ 2 that is ", math.pow(2,2))



print('################## Answer 2 E ##########################################') 
# At the second layer we will get the 4 leaves that are the final 4 nodes of decision tree
# for layer 2 left node split
# Here we check with with predictor attribute is having lowest entropy in the 
#left node as well as at the right node

#first check with left node 
########################################################################################
train_data_left_branch = claim_history2_train[claim_history2_train['OCCUPATION'].isin(subset1_occu)]

layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu = minimumOrdEnt(
    inData=train_data_left_branch[['EDUCATION', 'CAR_USE']], setIntervals=[0, 1, 2, 3, 4])
print(layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu)

layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = minimualNomEnt(
    inData=train_data_left_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type)

layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu = minimualNomEnt(
    inData=train_data_left_branch[['OCCUPATION', 'CAR_USE']], set=subset1_occu)
print(layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu)

#here we can see that Attribute = "Education" having lowest entropy among all three
# So we consider Education for next left node split

####################### Left: left node (1st leaf) ##########################################
train_data_left_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] <= layer1_interval_edu]


pri_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Private'].shape[0]
com_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Commercial'].shape[0]
total_count_l1 = train_data_left_left_branch.shape[0]

p_com_l1 = com_count_l1/total_count_l1
p_pri_l1 = pri_count_l1/total_count_l1

entropy_l1 = -((p_com_l1*math.log2(p_com_l1))+(p_pri_l1*math.log2(p_pri_l1)))
class_l1 = 'Commercial' if com_count_l1 > pri_count_l1 else 'Private'
print(entropy_l1, total_count_l1, com_count_l1, pri_count_l1, p_com_l1, p_pri_l1, class_l1)


#################### left : right node (2nd leaf) ##################################3
train_data_right_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] > layer1_interval_edu]

com_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Private'].shape[0]
total_count_l2 = train_data_right_left_branch.shape[0]

p_com_l2 = com_count_l2/total_count_l2
p_pri_l2 = pri_count_l2/total_count_l2

entropy_l2 = -((p_com_l2*math.log2(p_com_l2))+(p_pri_l2*math.log2(p_pri_l2)))
class_l2 = 'Commercial' if com_count_l2 > pri_count_l2 else 'Private'
print(entropy_l2, total_count_l2, com_count_l2, pri_count_l2, p_com_l2, p_pri_l2, class_l2)


#######################################################################################
#Check with right node
train_data_right_branch = claim_history2_train[claim_history2_train['OCCUPATION'].isin(subset2_occu)]

layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu = minimumOrdEnt(
    inData=train_data_right_branch[['EDUCATION', 'CAR_USE']], setIntervals=[0, 1, 2, 3, 4])
print(layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu)

layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = minimualNomEnt(
    inData=train_data_right_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type)

layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu = minimualNomEnt(
    inData=train_data_right_branch[['OCCUPATION', 'CAR_USE']], set=subset2_occu)
print(layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu)

#here we can see that Attribute = "CAR_TYPE" having lowest entropy among all three
# So we consider CAR_TYPE for next right node split
####################### Right:left node (3rd leaf) ##########################################

train_data_left_right_branch = train_data_right_branch[train_data_right_branch['CAR_TYPE'].isin(layer1_subset1_car_type)]

com_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Private'].shape[0]
total_count_l3 = train_data_left_right_branch.shape[0]

p_com_l3 = com_count_l3/total_count_l3
p_pri_l3 = pri_count_l3/total_count_l3
entropy_l3 = -((p_com_l3*math.log2(p_com_l3))+(p_pri_l3*math.log2(p_pri_l3)))
class_l3 = 'Commercial' if com_count_l3 > pri_count_l3 else 'Private'
print(entropy_l3, total_count_l3, com_count_l3, pri_count_l3, p_com_l3, p_pri_l3, class_l3)

####################### Right: Right node (4th leaf) ##########################################
train_data_right_right_branch = train_data_right_branch[train_data_right_branch['CAR_TYPE'].isin(layer1_subset2_car_type)]

com_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Private'].shape[0]
total_count_l4 = train_data_right_right_branch.shape[0]

p_com_l4 = com_count_l4/total_count_l4
p_pri_l4 = pri_count_l4/total_count_l4
entropy_l4 = -((p_com_l4*math.log2(p_com_l4))+(p_pri_l4*math.log2(p_pri_l4)))
class_l4 = 'Commercial' if com_count_l4 > pri_count_l4 else 'Private'
print(entropy_l4, total_count_l4, com_count_l4, pri_count_l4, p_com_l4, p_pri_l4, class_l4)




#print('################## Answer 2 F ##########################################') 

#First calculate Thresold value of event of Train partion
claim_history3 = claim_history[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()
claim_history3_train, claim_history3_test = train_test_split(claim_history, test_size = 0.25, random_state=60616, stratify = claim_history['CAR_USE'])

threshold = claim_history3_train.groupby("CAR_USE").size()["Commercial"] / claim_history3_train.shape[0]

#devide event and non-event by prediction probability

def predict_class(inData):
    if inData['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if inData['EDUCATION'] <= 0.5:
            return [0.2693548387096774]
        else:
            return [0.8376594808622966]
    else:
        if inData['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.008420441347270616]
        else:
            return [0.5341972642188625]


def predict_class_decision_tree(inData):
    out_data = numpy.ndarray(shape=(len(inData), 2), dtype=float)
    counter = 0
    for index, row in inData.iterrows():
        probability = predict_class(inData=row)
        out_data[counter] = probability
        counter += 1
    return out_data


## separate input and output variables in both part
train_data_x = claim_history3_train[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
train_data_y = claim_history3_train['CAR_USE']
test_data_x = claim_history3_test[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
test_data_y = claim_history3_test["CAR_USE"]

test_data_x['EDUCATION'] = test_data_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

train_data_x['EDUCATION'] = test_data_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})


# predict probability for testing input data
predProb_y = predict_class_decision_tree(inData=test_data_x)
predProb_y = predProb_y[:, 0] 

# determining the predicted class
pred_y = numpy.empty_like(test_data_y)
for i in range(test_data_y.shape[0]):
    if predProb_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'
 

print ("######################## Answer 2 F ######################################")
# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_data_y, predProb_y, pos_label = 'Commercial')

max = sys.float_info.min
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)

plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()

print(thresholds)
print(tpr-fpr)
print('The Kolmogorov-Smirnov statistic =',numpy.amax(tpr-fpr))
print('The event Probability cutoff =',thresholds[2])

print ("######################## Answer 3 A ######################################")
# calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(test_data_y, pred_y)
misclassification_rate = (1 - accuracy) * 100
print(f'The test partition Misclassification Rate: {misclassification_rate} %')

print ("######################## Answer 3 B ######################################")
# predict probability for training input data 
predProb_y_train = predict_class_decision_tree(inData=train_data_x)
predProb_y_train = predProb_y_train[:, 0] 
fpr, tpr, thresholds_new = metrics.roc_curve(train_data_y, predProb_y_train, pos_label = 'Commercial')
#
print(tpr-fpr)
print('The Kolmogorov-Smirnov statistic =',numpy.amax(tpr-fpr))
print('The event Probability cutoff =',thresholds_new[2])
# determining the predicted class
pred_y_train = numpy.empty_like(train_data_y)
for i in range(train_data_y.shape[0]):
    if predProb_y_train[i] > thresholds_new[2]:
        pred_y_train[i] = 'Commercial'
    else:
        pred_y_train[i] = 'Private'
        
accuracy = metrics.accuracy_score(train_data_y, pred_y_train)
misclassification_rate = (1 - accuracy) * 100
print(f'The test partition Misclassification Rate: {misclassification_rate} %')

       
print ("######################## Answer 3 C ######################################")
# Root Average Squared Error in the Test partition
RASE = 0.0
lastindex = test_data_y.shape[0]
i = 0
for i in range(0,lastindex):
    if test_data_y.iloc[i] == 'Commercial':
        RASE += (1 - predProb_y[i])**2
    else:
        RASE += (0 - predProb_y[i])**2
RASE = numpy.sqrt(RASE/test_data_y.shape[0])
print("Root Average Square Error is ",RASE)


print ("######################## Answer 3 D ######################################")
# AUC of Test Partion
y_true = 1.0 * numpy.isin(test_data_y, ['Commercial'])
AUC = metrics.roc_auc_score(y_true, predProb_y)
print("Area Under Curve for Test Partion", AUC)

print ("######################## Answer 3 E ######################################")
#The Gini Coefficient in the Test partition

Gini_Coefficient = (AUC - 0.5)/0.5
print("Gini Coefficient in the Test partition =",Gini_Coefficient )      
       
print ("######################## Answer 3 F ######################################")
#The Goodman-Kruskal Gamma statistic in the Test partition
df = pd.DataFrame(test_data_y)
df['ppr'] = predProb_y.tolist()
gk = df.groupby('CAR_USE')
col_data = gk.get_group("Private").sort_values(by=['ppr']) # Non- event Probability
row_data = gk.get_group("Commercial").sort_values(by=['ppr']) # event probability
#row_data.iloc[0,1]  # 0 = row , 1 = column
ties_pair = 0
Concordant_pair = 0
Discordant_pair = 0
for i in range(0,col_data.shape[0]):
    for j in range(0,row_data.shape[0]):
        if float(row_data.iloc[j,1]) == float(col_data.iloc[i,1]):
            ties_pair = ties_pair + 1
        elif float(row_data.iloc[j,1]) < float(col_data.iloc[i,1]):
            Discordant_pair = Discordant_pair + 1
        elif float(row_data.iloc[j,1]) > float(col_data.iloc[i,1]):
            Concordant_pair = Concordant_pair + 1
            
#print (ties_pair)
#print (Concordant_pair)
#print (Discordant_pair)
GK_statistic = (Concordant_pair - Discordant_pair)/ (Concordant_pair + Discordant_pair)
print("The Goodman-Kruskal Gamma statistic in the Test partition = ",GK_statistic )

print ("######################## Answer 3 G ######################################")
# Generate ROC Curve
# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(test_data_y, predProb_y, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], fpr)
Sensitivity = numpy.append([0], tpr)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'green', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.title(label = "ROC Curve for Test Partition")
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
plt.show()

############################## END ##################################################