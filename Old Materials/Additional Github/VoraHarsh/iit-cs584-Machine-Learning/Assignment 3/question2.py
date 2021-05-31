import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from itertools import combinations
import math
from sklearn import metrics
import matplotlib.pyplot as plt

def EntropyOrdinalSplit(in_data, split):
    data_table = in_data
    data_table['LE_Split'] = (data_table.iloc[:,0] <= split)

    cross_table = pd.crosstab(index=data_table['LE_Split'], columns=data_table.iloc[:, 1], margins=True, dropna=True)
    n_rows = cross_table.shape[0]
    n_columns = cross_table.shape[1]

    table_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
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

    n_rows = cross_table.shape[0]
    n_columns = cross_table.shape[1]

    table_entropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = cross_table.iloc[i_row, i_column] / cross_table.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * np.log2(proportion)
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
        table, entropy = EntropyNominalSplit(in_data=in_data, subset=subset)
        if entropy < min_entropy:
            min_entropy = entropy
            min_subset1 = subset
            min_subset2 = subset_map.get(subset)
            min_table = table

    return min_table, min_entropy, min_subset1, min_subset2

dataframe = pd.read_csv('claim_history.csv', delimiter=',')
dataframe = dataframe[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]]

dataframe['EDUCATION'] = dataframe['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

X = dataframe[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]


training_data = 0.75
testing_data = 0.25
train_data, test_data = train_test_split(dataframe, train_size=training_data, test_size=testing_data, random_state=60616, stratify=dataframe["CAR_USE"])

prob_comm_training = train_data.groupby('CAR_USE').size()['Commercial'] / train_data.shape[0]
prob_priv_trainining = train_data.groupby('CAR_USE').size()['Private'] / train_data.shape[0]

#Question 1

print("(5 points). What is the entropy value of the root node?")
root_entropy = -((prob_comm_training * math.log2(prob_comm_training)) + (prob_priv_trainining * math.log2(prob_priv_trainining)))
print("Root Node Entropy value:", root_entropy)
print()
print("="*50)

#Question 2
print("(5 points). What is the split criterion (i.e., predictor name and values in the two branches) of the first layer?")
cross_table_edu, entropy_table_edu, interval_edu = FindMinOrdinalEntropy(in_data=train_data[['EDUCATION', 'CAR_USE']],set_intervals=[0, 1, 2, 3, 4])
print("Cross Table for Education\n", cross_table_edu)
print("Entropy for Education", entropy_table_edu)
print("Split Interval", interval_edu)
print()

cross_table_car_type, table_entropy_car_type, subset1_car_type, subset2_car_type = FindMinNominalEntropy(in_data=train_data[['CAR_TYPE', 'CAR_USE']], set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print("Cross Table for Car-Type\n", cross_table_car_type)
print("Entropy for Car-Type", table_entropy_car_type)
print("Left Subset", subset1_car_type)
print("Right Subset", subset2_car_type)
print()

cross_table_occu, table_entropy_occu, subset1_occu, subset2_occu = FindMinNominalEntropy(
    in_data=train_data[['OCCUPATION', 'CAR_USE']],set=['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
print("Cross Table for Occupation\n", cross_table_occu)
print("Entropy for Occupation", table_entropy_occu)
print("Left Subset", subset1_occu)
print("Right Subset", subset2_occu)
print()

print("Split criterion for first layer")
print("predictor name: OCCUPATION")
print("predictor value: ")
print("Left Subset: ", subset1_occu)
print("Right Subset: ", subset2_occu)
print()

print("="*50)

#Question 3
print("(10 points). What is the entropy of the split of the first layer?")

# for layer 1 left node split
train_data_left_branch = train_data[train_data['OCCUPATION'].isin(subset1_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu_l1 = FindMinOrdinalEntropy(
    in_data=train_data_left_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print("Left Node Split")
print("Cross Table for Education\n", layer1_cross_table_edu)
print("Entropy for Education", layer1_table_entropy_edu)
print("Split Interval", layer1_interval_edu_l1)
print()

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = FindMinNominalEntropy(in_data=train_data_left_branch[['CAR_TYPE', 'CAR_USE']],set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print("Left Node Split")
print("Cross Table for Car-Type\n", layer1_cross_table_car_type)
print("Entropy for Car-Type", layer1_table_entropy_car_type)
print("Left Subset", layer1_subset1_car_type)
print("Right Subset", layer1_subset2_car_type)
print()

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(
    in_data=train_data_left_branch[['OCCUPATION', 'CAR_USE']], set=subset1_occu)
print("Left Node Split")
print("Cross Table for Occupation\n", layer1_cross_table_occu)
print("Entropy for Occupation", layer1_table_entropy_occu)
print("Left Subset", layer1_subset1_occu)
print("Right Subset", layer1_subset2_occu)
print()


# for layer 1 right node split
train_data_right_branch = train_data[train_data['OCCUPATION'].isin(subset2_occu)]

layer1_cross_table_edu, layer1_table_entropy_edu, layer1_interval_edu = FindMinOrdinalEntropy(in_data=train_data_right_branch[['EDUCATION', 'CAR_USE']], set_intervals=[0, 1, 2, 3, 4])
print("Right Node Split")
print("Cross Table for Education\n", layer1_cross_table_edu)
print("Entropy for Education", layer1_table_entropy_edu)
print("Split Interval", layer1_interval_edu)
print()

layer1_cross_table_car_type, layer1_table_entropy_car_type, layer1_subset1_car_type_l1, layer1_subset2_car_type_l1 = FindMinNominalEntropy(in_data=train_data_right_branch[['CAR_TYPE', 'CAR_USE']],set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print("Right Node Split")
print("Cross Table for Car-Type\n", layer1_cross_table_car_type)
print("Entropy for Car-Type", layer1_table_entropy_car_type)
print("Left Subset", layer1_subset1_car_type_l1)
print("Right Subset", layer1_subset2_car_type_l1)
print()

layer1_cross_table_occu, layer1_table_entropy_occu, layer1_subset1_occu, layer1_subset2_occu = FindMinNominalEntropy(in_data=train_data_right_branch[['OCCUPATION', 'CAR_USE']], set=subset2_occu)
print("Right Node Split")
print("Cross Table for Occupation\n", layer1_cross_table_occu)
print("Entropy for Occupation", layer1_table_entropy_occu)
print("Left Subset", layer1_subset1_occu)
print("Right Subset", layer1_subset2_occu)
print()

print("Entropy of the split of the first layer:")
print("Entropy:",table_entropy_occu)
print("CrossTable:\n",cross_table_occu)

print("="*50)

#Question 4
print("(5 points). How many leaves?")
print('There are 4 leaves')
print("="*50)

#Question 5

print("(10 points). Describe all your leaves.  Please include the decision rules and the counts of the target values")

#data of leaf 1
train_data_left_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] <= layer1_interval_edu_l1]
total_count_l1 = train_data_left_left_branch.shape[0]
com_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l1 = train_data_left_left_branch[train_data_left_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l1 = com_count_l1 / total_count_l1
p_pri_l1 = pri_count_l1 / total_count_l1
entropy_l1 = -((p_com_l1 * math.log2(p_com_l1)) + (p_pri_l1 * math.log2(p_pri_l1)))
class_l1 = 'Commercial' if com_count_l1 > pri_count_l1 else 'Private'
print("Leaf 1:")
print('Decision rules are 1.{} and 2.{}: '.format(subset1_occu,['Below High School']))
print("Entropy: ",entropy_l1)
print("Total Count: ",total_count_l1)
print("Commercial Count: ", com_count_l1)
print("Private Count: ", pri_count_l1)
print("Commercial Probability: ",p_com_l1)
print("Private Probability:",p_pri_l1)
print("Class: ", class_l1)
print()

#data of leaf 2
train_data_right_left_branch = train_data_left_branch[train_data_left_branch['EDUCATION'] > layer1_interval_edu_l1]
total_count_l2 = train_data_right_left_branch.shape[0]
com_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l2 = train_data_right_left_branch[train_data_right_left_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l2 = com_count_l2 / total_count_l2
p_pri_l2 = pri_count_l2 / total_count_l2
entropy_l2 = -((p_com_l2 * math.log2(p_com_l2)) + (p_pri_l2 * math.log2(p_pri_l2)))
class_l2 = 'Commercial' if com_count_l2 > pri_count_l2 else 'Private'
print("Leaf 2:")
print('Decision rules are 1.{} and 2.{}: '.format(subset1_occu,['High School', 'Bachelors', 'Masters', 'Doctors']))
print("Entropy: ",entropy_l2)
print("Total Count: ",total_count_l2)
print("Commercial Count: ", com_count_l2)
print("Private Count: ", pri_count_l2)
print("Commercial Probability: ",p_com_l2)
print("Private Probability:",p_pri_l2)
print("Class: ", class_l2)
print()

# data of leaf 3
train_data_left_right_branch = train_data_right_branch[train_data_right_branch['CAR_TYPE'].isin(layer1_subset1_car_type_l1)]
total_count_l3 = train_data_left_right_branch.shape[0]
com_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l3 = train_data_left_right_branch[train_data_left_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l3 = com_count_l3 / total_count_l3
p_pri_l3 = pri_count_l3 / total_count_l3
entropy_l3 = -((p_com_l3 * math.log2(p_com_l3)) + (p_pri_l3 * math.log2(p_pri_l3)))
class_l3 = 'Commercial' if com_count_l3 > pri_count_l3 else 'Private'
print("Leaf 3:")
print('Decision rules are 1.{} and 2.{}: '.format(subset2_occu,layer1_subset1_car_type_l1))
print("Entropy: ",entropy_l3)
print("Total Count: ",total_count_l3)
print("Commercial Count: ", com_count_l3)
print("Private Count: ", pri_count_l3)
print("Commercial Probability: ",p_com_l3)
print("Private Probability:",p_pri_l3)
print("Class: ", class_l3)
print()


# data of leave 4
train_data_right_right_branch = train_data_right_branch[train_data_right_branch['CAR_TYPE'].isin(layer1_subset2_car_type_l1)]
total_count_l4 = train_data_right_right_branch.shape[0]
com_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Commercial'].shape[0]
pri_count_l4 = train_data_right_right_branch[train_data_right_right_branch['CAR_USE'] == 'Private'].shape[0]
p_com_l4 = com_count_l4 / total_count_l4
p_pri_l4 = pri_count_l4 / total_count_l4
entropy_l4 = -((p_com_l4 * math.log2(p_com_l4)) + (p_pri_l4 * math.log2(p_pri_l4)))
class_l4 = 'Commercial' if com_count_l4 > pri_count_l4 else 'Private'
print("Leaf 4:")
print('Decision rules are 1.{} and 2.{}: '.format(subset2_occu,layer1_subset2_car_type_l1))
print("Entropy: ",entropy_l4)
print("Total Count: ",total_count_l4)
print("Commercial Count: ", com_count_l4)
print("Private Count: ", pri_count_l4)
print("Commercial Probability: ",p_com_l4)
print("Private Probability:",p_pri_l4)
print("Class: ", class_l4)
print()

#Question 5

print("(5 points). What are the Kolmogorov-Smirnov statistic and the event probability cutoff value?")

def predict_class(data):
    if data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if data['EDUCATION'] <= 0.5:
            return [0.2693548387096774, 0.7306451612903225]
        else:
            return [0.8376594808622966, 0.16234051913770348]
    else:
        if data['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.008420441347270616, 0.9915795586527294]
        else:
            return [0.5341972642188625, 0.4658027357811375]

def decisiontree(data):
    outputdata = np.ndarray(shape=(len(data), 2), dtype=float)
    count = 0
    for index, row in data.iterrows():
        prob = predict_class(data=row)
        outputdata[count] = prob
        count += 1
    return outputdata

train_x, test_x, train_y, test_y = train_test_split(X, dataframe["CAR_USE"],train_size=training_data, test_size=testing_data, random_state=60616, stratify=dataframe["CAR_USE"])
train_x['EDUCATION_VAL'] = train_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
test_x['EDUCATION_VAL'] = test_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
threshold = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]
pred_prob_y = decisiontree(data=test_x)
pred_prob_y = pred_prob_y[:, 0]
num_y = test_y.shape[0]
pred_y = np.empty_like(test_y)

for i in range(num_y):
    if pred_prob_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'

falsepositive, truepositive, thresholds = metrics.roc_curve(test_y, pred_prob_y, pos_label='Commercial')
cutoff = np.where(thresholds > 1.0, np.nan, thresholds)
plt.plot(cutoff, truepositive, marker='o', label='True Positive',
         color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot(cutoff, falsepositive, marker='o', label='False Positive',
         color='orange', linestyle='solid', linewidth=2, markersize=6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.show()

ksStatistic = 0
ksThreshold = 0
for i in range(len(thresholds)):
    if truepositive[i]-falsepositive[i]>ksStatistic:
        ksStatistic = truepositive[i]-falsepositive[i]
        ksThreshold = thresholds[i]
print("The Kolmogorov-Smirnov statistic is ",ksStatistic)
print("Event probability cutoff value",ksThreshold)


