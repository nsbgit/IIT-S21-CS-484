#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[1]:


# load necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Please provide information about your Data Partition step.

# load data of claim history
claim_history_data = pd.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 3\\claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()
data_shape = claim_history_data.shape
print(f'\nclaim history data :\n{claim_history_data}')
print(f'claim history data shape : {data_shape}')

# take partition of data into training and testing data
p_training = 0.7
p_testing = 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing,
                                                                     random_state=27513, stratify=claim_history_data["CAR_USE"])

train_data = claim_history_data_train
test_data = claim_history_data_test
print('number of Observations in training data = ', train_data.shape[0])
print('number of Observations in testing data = ', test_data.shape[0])


# # a)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Training partition?

# In[2]:


print(f'count of target variable in train data :\n{train_data.groupby("CAR_USE").size()}')
print(f'proportion of target variable in train data :\n {train_data.groupby("CAR_USE").size() / train_data.shape[0]}')


# # b)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Test partition?

# In[3]:


print(f'count of target variable in test data :\n{test_data.groupby("CAR_USE").size()}')
print(f'proportion of target variable in test data :\n {test_data.groupby("CAR_USE").size() / test_data.shape[0]}')


# # c)	(5 points). What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?

# In[4]:


p_com_given_training = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]
p_com_given_testing = test_data.groupby("CAR_USE").size()["Commercial"] / test_data.shape[0]
p_com = (p_com_given_training * p_training) + (p_com_given_testing * p_testing)
p_training_given_com = (p_com_given_training * p_training) / p_com
print(
    f'probability that an observation is in the Training partition given that CAR_USE = Commercial : {p_training_given_com}')


# # d)	(5 points). What is the probability that an observation is in the Test partition given that CAR_USE = Private?

# In[5]:


p_pri_given_testing = test_data.groupby("CAR_USE").size()["Private"] / test_data.shape[0]
p_pri_given_training = train_data.groupby("CAR_USE").size()["Private"] / train_data.shape[0]
p_pri = (p_pri_given_testing * p_testing) + (p_pri_given_training * p_training)
p_testing_given_pri = (p_pri_given_testing * p_testing) / p_pri
print(f'probability that an observation is in the Test partition given that CAR_USE = Private : {p_testing_given_pri}')


# # Question 2

# In[6]:


# load necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from itertools import combinations
import math
import warnings
warnings.filterwarnings("ignore")


# function that calculate entropy for ordinal or interval variables
def EntropyOrdinalSplit(in_data, split):
    data_table = in_data
    data_table['LE_Split'] = (data_table.iloc[:, 0] <= split)

    cross_table = pd.crosstab(index=data_table['LE_Split'], columns=data_table.iloc[:, 1], margins=True, dropna=True)
    # print(cross_table)

    n_rows = cross_table.shape[0]
    n_columns = cross_table.shape[1]

    table_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
        # print('Row = ', i_row, 'Entropy =', row_entropy)
        # print(' ')
        table_entropy += row_entropy * cross_table.iloc[i_row, (n_columns - 1)]
    table_entropy = table_entropy / cross_table.iloc[(n_rows - 1), (n_columns - 1)]

    return cross_table, table_entropy


# function that check all the posibilities in ordinal or interval variables and return possibility that has smallest
# entropy
def FindMinOrdinalEntropy(in_data, set_intervals):
    min_entropy = sys.float_info.max
    min_interval = None
    min_table = None

    for i in range(set_intervals[0], set_intervals[len(set_intervals) - 1]):
        ret_table, ret_entropy = EntropyOrdinalSplit(in_data=in_data, split=i + 0.5)
        if ret_entropy < min_entropy:
            min_entropy = ret_entropy
            min_interval = i + 0.5
            min_table = ret_table

    return min_table, min_entropy, min_interval


# function that calculate entropy for nominal variables
def EntropyNominalSplit(in_data, subset):
    data_table = in_data
    data_table['LE_Split'] = data_table.iloc[:, 0].apply(lambda x: True if x in subset else False)

    cross_table = pd.crosstab(index=data_table['LE_Split'], columns=data_table.iloc[:, 1], margins=True, dropna=True)
    # print(cross_table)

    n_rows = cross_table.shape[0]
    n_columns = cross_table.shape[1]

    table_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
        # print('Row = ', i_row, 'Entropy =', row_entropy)
        # print(' ')
        table_entropy += row_entropy * cross_table.iloc[i_row, (n_columns - 1)]
    table_entropy = table_entropy / cross_table.iloc[(n_rows - 1), (n_columns - 1)]

    return cross_table, table_entropy


# function that create all the possible combinations for nominal variables and return possibility that has smallest
# entropy
def FindMinNominalEntropy(in_data, set):
    subset_map = {}
    for i in range(1, (int(len(set) / 2)) + 1):
        subsets = combinations(set, i)
        for ss in subsets:
            remaining = tuple()
            for ele in set:
                if ele not in ss:
                    remaining += (ele,)
            if subset_map.get(remaining) == None:
                subset_map[ss] = remaining

    min_entropy = sys.float_info.max
    min_subset1 = None
    min_subset2 = None
    min_table = None

    for subset in subset_map:
        ret_table, ret_entropy = EntropyNominalSplit(in_data=in_data, subset=subset)
        if ret_entropy < min_entropy:
            min_entropy = ret_entropy
            min_subset1 = subset
            min_subset2 = subset_map.get(subset)
            min_table = ret_table

    return min_table, min_entropy, min_subset1, min_subset2

# loading data from data file
claim_history_data = pd.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 3\\claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()
data_shape = claim_history_data.shape

# map ordinal variable with numeric data
claim_history_data['EDUCATION'] = claim_history_data['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
# print(claim_history_data)

# splitting data into training and testing part
p_training, p_testing = 0.7, 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing, random_state=27513)
train_data = claim_history_data_train
test_data = claim_history_data_test


# # a)	(5 points). What is the entropy value of the root node?

# In[9]:


p_com_train = train_data.groupby('CAR_USE').size()['Commercial'] / train_data.shape[0]
p_pri_train = train_data.groupby('CAR_USE').size()['Private'] / train_data.shape[0]

root_entropy = -((p_com_train * math.log2(p_com_train)) + (p_pri_train * math.log2(p_pri_train)))
print(f'root node entropy : {root_entropy}')


# # b)	(5 points). What is the split criterion (i.e., predictor name and values in the two branches) of the first layer?

# In[10]:


# for layer 0 split
cross_table_edu, table_entropy_edu, interval_edu = FindMinOrdinalEntropy(in_data=train_data[['EDUCATION', 'CAR_USE']],
                                                                         set_intervals=[0, 1, 2, 3, 4])
print(
    f'layer0-education:\n '
    f'cross table: \n{cross_table_edu}\n '
    f'entropy: {table_entropy_edu}\n '
    f'split interval: {interval_edu}\n')

cross_table_car_type, table_entropy_car_type, subset1_car_type, subset2_car_type = FindMinNominalEntropy(
    in_data=train_data[['CAR_TYPE', 'CAR_USE']], set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(
    f'layer0-car-type:\n '
    f'cross table: \n{cross_table_car_type}\n '
    f'entropy: {table_entropy_car_type}\n '
    f'left subset: {subset1_car_type}\n '
    f'right subset: {subset2_car_type}\n')

cross_table_occu, table_entropy_occu, subset1_occu, subset2_occu = FindMinNominalEntropy(
    in_data=train_data[['OCCUPATION', 'CAR_USE']],
    set=['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
print(
    f'layer0-occupation:\n '
    f'cross table: \n{cross_table_occu}\n '
    f'entropy: {table_entropy_occu}\n '
    f'left subset: {subset1_occu}\n '
    f'right subset: {subset2_occu}\n')

print(f'split criterion for first layer')
print(f'predictor name: OCCUPATION')
print(f'predictor value:\n left subset: {subset1_occu}\n right subset: {subset2_occu}')


# # c)	(10 points). What is the entropy of the split of the first layer?

# In[11]:


# for layer 1 left node split
train_data_left_branch = train_data[train_data['OCCUPATION'].isin(subset1_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(
    in_data=train_data_left_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print(
    f'layer1-left-node-education:\n cross table: \n{layer1_cross_table_edu}\n entropy: {layer1_table_entropy_edu}\n '
    f'split interval: {layer1_interval_edu}\n')

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(
    in_data=train_data_left_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(
    f'layer1-left-node-car-type:\n '
    f'cross table: \n{layer1_cross_table_car_type}\n '
    f'entropy: {layer1_table_entropy_car_type}\n '
    f'left subset: {layer1_subset1_car_type}\n '
    f'right subset: {layer1_subset2_car_type}\n')

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_left_branch[['OCCUPATION', 'CAR_USE']], set=subset1_occu)
print(
    f'layer1-left-node-occupation:\n '
    f'cross table: \n{layer1_cross_table_occu}\n '
    f'entropy: {layer1_table_entropy_occu}\n '
    f'left subset: {layer1_subset1_occu}\n '
    f'right subset: {layer1_subset2_occu}\n')

# for layer 1 right node split
train_data_right_branch = train_data[train_data['OCCUPATION'].isin(subset2_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(
    in_data=train_data_right_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print(
    f'layer1-right-node-education:\n '
    f'cross table: \n{layer1_cross_table_edu}\n '
    f'entropy: {layer1_table_entropy_edu}\n '
    f'split interval: {layer1_interval_edu}\n')

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(
    in_data=train_data_right_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(
    f'layer1-right-node-car-type:\n '
    f'cross table: \n{layer1_cross_table_car_type}\n '
    f'entropy: {layer1_table_entropy_car_type}\n '
    f'left subset: {layer1_subset1_car_type}\n '
    f'right subset: {layer1_subset2_car_type}\n')

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_right_branch[['OCCUPATION', 'CAR_USE']], set=subset2_occu)
print(
    f'layer1-left-node-occupation:\n '
    f'cross table: \n{layer1_cross_table_occu}\n '
    f'entropy: {layer1_table_entropy_occu}\n '
    f'left subset: {layer1_subset1_occu}\n '
    f'right subset: {layer1_subset2_occu}\n')

print(
    f'entropy of the split of the first layer:\n '
    f'for left node: {layer1_table_entropy_edu}\n '
    f'for right node: {layer1_table_entropy_car_type}\n')


# # d)	(5 points). How many leaves?

# In[16]:


print('There are 4 leaves')


# # e)	(15 points). Describe all your leaves.  Please include the decision rules and the counts of the target values.

# In[17]:


# data of leave 1
train_data_left_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] <= layer1_interval_edu]
total_count_l1 = train_data_left_left_branch.shape[0]
com_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l1 = com_count_l1 / total_count_l1
p_pri_l1 = pri_count_l1 / total_count_l1
entropy_l1 = -((p_com_l1 * math.log2(p_com_l1)) + (p_pri_l1 * math.log2(p_pri_l1)))
class_l1 = 'Commercial' if com_count_l1 > pri_count_l1 else 'Private'
print(
    f'leave 1:\n entropy: {entropy_l1}\n total count: {total_count_l1}\n commercial count: {com_count_l1}\n '
    f'private count: {pri_count_l1}\n commercial probability: {p_com_l1}\n private probability: {p_pri_l1}\n '
    f'class: {class_l1}\n')

# data of leave 2
train_data_right_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] > layer1_interval_edu]
total_count_l2 = train_data_right_left_branch.shape[0]
com_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l2 = com_count_l2 / total_count_l2
p_pri_l2 = pri_count_l2 / total_count_l2
entropy_l2 = -((p_com_l2 * math.log2(p_com_l2)) + (p_pri_l2 * math.log2(p_pri_l2)))
class_l2 = 'Commercial' if com_count_l2 > pri_count_l2 else 'Private'
print(
    f'leave 2:\n entropy: {entropy_l2}\n total count: {total_count_l2}\n commercial count: {com_count_l2}\n '
    f'private count: {pri_count_l2}\n commercial probability: {p_com_l2}\n private probability: {p_pri_l2}\n '
    f'class: {class_l2}\n')

# data of leave 3
train_data_left_right_branch = train_data_right_branch[
    train_data_right_branch['CAR_TYPE'].isin(layer1_subset1_car_type)]
total_count_l3 = train_data_left_right_branch.shape[0]
com_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l3 = com_count_l3 / total_count_l3
p_pri_l3 = pri_count_l3 / total_count_l3
entropy_l3 = -((p_com_l3 * math.log2(p_com_l3)) + (p_pri_l3 * math.log2(p_pri_l3)))
class_l3 = 'Commercial' if com_count_l3 > pri_count_l3 else 'Private'
print(
    f'leave 3:\n entropy: {entropy_l3}\n total count: {total_count_l3}\n commercial count: {com_count_l3}\n '
    f'private count: {pri_count_l3}\n commercial probability: {p_com_l3}\n private probability: {p_pri_l3}\n '
    f'class: {class_l3}\n')

# data of leave 4
train_data_right_right_branch = train_data_right_branch[
    train_data_right_branch['CAR_TYPE'].isin(layer1_subset2_car_type)]
total_count_l4 = train_data_right_right_branch.shape[0]
com_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l4 = com_count_l4 / total_count_l4
p_pri_l4 = pri_count_l4 / total_count_l4
entropy_l4 = -((p_com_l4 * math.log2(p_com_l4)) + (p_pri_l4 * math.log2(p_pri_l4)))
class_l4 = 'Commercial' if com_count_l4 > pri_count_l4 else 'Private'
print(
    f'leave 4:\n entropy: {entropy_l4}\n total count: {total_count_l4}\n commercial count: {com_count_l4}\n '
    f'private count: {pri_count_l4}\n commercial probability: {p_com_l4}\n private probability: {p_pri_l4}\n '
    f'class: {class_l4}\n')


# # Question 3

# In[18]:


# load necessary libraries
import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as metrics
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# function based decision tree to predict class probability
def predict_class(in_data):
    if in_data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if in_data['EDUCATION'] <= 0.5:
            return [0.6832518880497557, 0.3167481119502443]
        else:
            return [0.8912579957356077, 0.10874200426439233]
    else:
        if in_data['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.006838669567920423, 0.9931613304320795]
        else:
            return [0.5306122448979592, 0.46938775510204084]


# function that return probability of all inputted data
def predict_class_decision_tree(in_data):
    out_data = numpy.ndarray(shape=(len(in_data), 2), dtype=float)
    counter = 0
    for index, row in in_data.iterrows():
        probability = predict_class(in_data=row)
        out_data[counter] = probability
        counter += 1
    return out_data


# In[19]:


# Please apply your decision tree to the Test partition and then provide the following information.

# loading data from data file
claim_history_data = pd.read_csv('C:\\Users\\Machine Learning\\Assignments & Projects\\Assignment 3\\claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()

# splitting data into training and testing part
p_training, p_testing = 0.7, 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing, random_state=27513)
train_data = claim_history_data_train
test_data = claim_history_data_test
# separate input and output variables in both part
train_data_x = train_data[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
train_data_y = train_data['CAR_USE']
test_data_x = test_data[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
test_data_y = test_data["CAR_USE"]

test_data_x['EDUCATION'] = test_data_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})


# # a)	(10 points). Use the proportion of target Event value in the training partition as the threshold, what is the misclassification Rate in the Test partition?

# In[20]:


# calculating threshold
threshold = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]
print(f'threshold is {threshold}')

# predict probability for testing input data
pred_prob_y = predict_class_decision_tree(in_data=test_data_x)
pred_prob_y = pred_prob_y[:, 0]

target_y = test_data_y
num_y = target_y.shape[0]

# determining the predicted class
pred_y = numpy.empty_like(target_y)
for i in range(num_y):
    if pred_prob_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'

# calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(target_y, pred_y)
misclassification_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')


# # b)     (10 points). What is the Root Average Squared Error in the Test partition?

# In[21]:


# calculating the root average squared error by applying the formula
RASE = 0.0
for y, ppy in zip(target_y, pred_prob_y):
    if y == 'Commercial':
        RASE += (1 - ppy) ** 2
    else:
        RASE += (0 - ppy) ** 2
RASE = numpy.sqrt(RASE / num_y)
print(f'Root Average Squared Error: {RASE}')


# # c)    (10 points). What is the Area Under Curve in the Test partition?

# In[22]:


# calculating the area under curve by applying the formula
y_true = 1.0 * numpy.isin(target_y, ['Commercial'])
AUC = metrics.roc_auc_score(y_true, pred_prob_y)
print(f'Area Under Curve: {AUC}')


# # d)	(10 points). Generate the Receiver Operating Characteristic curve for the Test partition.  The axes must be properly labeled.  Also, don’t forget the diagonal reference line.

# In[23]:


# generating the coordinates for the roc curve
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(target_y, pred_prob_y, pos_label='Commercial')

# adding two dummy coordinates
one_minus_specificity = numpy.append([0], one_minus_specificity)
sensitivity = numpy.append([0], sensitivity)

one_minus_specificity = numpy.append(one_minus_specificity, [1])
sensitivity = numpy.append(sensitivity, [1])

# plotting the roc curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

