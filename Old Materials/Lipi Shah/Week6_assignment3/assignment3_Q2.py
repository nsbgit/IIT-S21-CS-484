import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from itertools import combinations
import math


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


print("=" * 50)
print("=" * 50)
print("ML-Assignment 3-Question 2")
print("=" * 50)
print("=" * 50)

# loading data from data file
claim_history_data = pd.read_csv('claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()
data_shape = claim_history_data.shape

claim_history_data['EDUCATION'] = claim_history_data['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
print(claim_history_data)

# splitting data into training and testing part
p_training, p_testing = 0.7, 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing, random_state=27513)
train_data = claim_history_data_train
test_data = claim_history_data_test

p_com_train = train_data.groupby('CAR_USE').size()['Commercial'] / train_data.shape[0]
p_pri_train = train_data.groupby('CAR_USE').size()['Private'] / train_data.shape[0]

root_entropy = -((p_com_train * math.log2(p_com_train)) + (p_pri_train * math.log2(p_pri_train)))
print(f'root node entropy : {root_entropy}')

# for layer 0 split
cross_table_edu, table_entropy_edu, interval_edu = FindMinOrdinalEntropy(in_data=train_data[['EDUCATION', 'CAR_USE']],
                                                                         set_intervals=[0, 1, 2, 3, 4])
print(cross_table_edu, table_entropy_edu, interval_edu)

cross_table_car_type, table_entropy_car_type, subset1_car_type, subset2_car_type = FindMinNominalEntropy(
    in_data=train_data[['CAR_TYPE', 'CAR_USE']], set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(cross_table_car_type, table_entropy_car_type, subset1_car_type, subset2_car_type)

cross_table_occu, table_entropy_occu, subset1_occu, subset2_occu = FindMinNominalEntropy(
    in_data=train_data[['OCCUPATION', 'CAR_USE']],
    set=['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
print(cross_table_occu, table_entropy_occu, subset1_occu, subset2_occu)

# for layer 1 left node split
train_data_left_branch = train_data[train_data['OCCUPATION'].isin(subset1_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(
    in_data=train_data_left_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print(layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu)

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(
    in_data=train_data_left_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type)

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_left_branch[['OCCUPATION', 'CAR_USE']], set=subset1_occu)
print(layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu)

# for layer 1 right node split
train_data_right_branch = train_data[train_data['OCCUPATION'].isin(subset2_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(
    in_data=train_data_right_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print(layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu)

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(
    in_data=train_data_right_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type)

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_right_branch[['OCCUPATION', 'CAR_USE']], set=subset2_occu)
print(layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu)

train_data_left_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] <= layer1_interval_edu]
total_count_l1 = train_data_left_left_branch.shape[0]
com_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l1 = com_count_l1/total_count_l1
p_pri_l1 = pri_count_l1/total_count_l1
entropy_l1 = -((p_com_l1*math.log2(p_com_l1))+(p_pri_l1*math.log2(p_pri_l1)))
class_l1 = 'Commercial' if com_count_l1 > pri_count_l1 else 'Private'
print(entropy_l1, total_count_l1, com_count_l1, pri_count_l1, p_com_l1, p_pri_l1, class_l1)

train_data_right_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] > layer1_interval_edu]
total_count_l2 = train_data_right_left_branch.shape[0]
com_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l2 = com_count_l2/total_count_l2
p_pri_l2 = pri_count_l2/total_count_l2
entropy_l2 = -((p_com_l2*math.log2(p_com_l2))+(p_pri_l2*math.log2(p_pri_l2)))
class_l2 = 'Commercial' if com_count_l2 > pri_count_l2 else 'Private'
print(entropy_l2, total_count_l2, com_count_l2, pri_count_l2, p_com_l2, p_pri_l2, class_l2)

train_data_left_right_branch = train_data_right_branch[train_data_right_branch['CAR_TYPE'].isin(layer1_subset1_car_type)]
total_count_l3 = train_data_left_right_branch.shape[0]
com_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l3 = com_count_l3/total_count_l3
p_pri_l3 = pri_count_l3/total_count_l3
entropy_l3 = -((p_com_l3*math.log2(p_com_l3))+(p_pri_l3*math.log2(p_pri_l3)))
class_l3 = 'Commercial' if com_count_l3 > pri_count_l3 else 'Private'
print(entropy_l3, total_count_l3, com_count_l3, pri_count_l3, p_com_l3, p_pri_l3, class_l3)

train_data_right_right_branch = train_data_right_branch[train_data_right_branch['CAR_TYPE'].isin(layer1_subset2_car_type)]
total_count_l4 = train_data_right_right_branch.shape[0]
com_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l4 = com_count_l4/total_count_l4
p_pri_l4 = pri_count_l4/total_count_l4
entropy_l4 = -((p_com_l4*math.log2(p_com_l4))+(p_pri_l4*math.log2(p_pri_l4)))
class_l4 = 'Commercial' if com_count_l4 > pri_count_l4 else 'Private'
print(entropy_l4, total_count_l4, com_count_l4, pri_count_l4, p_com_l4, p_pri_l4, class_l4)
print(train_data.shape[0])

print()
print("=" * 50)
print("ML-Assignment 3-Question 2-Section a)")
print("=" * 50)
