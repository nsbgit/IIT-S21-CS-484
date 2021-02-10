
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import sys
from itertools import combinations
import math


claim_history = pandas.read_csv('D:\\IIT Edu\\Sem1\\MachineLearning\\Week6_assignment3\\claim_history.csv',delimiter=',')
X = claim_history.iloc[:, [11,12,17]].values
Y = claim_history.iloc[:, 14].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 27513, stratify = claim_history['CAR_USE'])

print('*************************Question 1 (a)*************************\n')

print('Count of the target variable in the Training partition :\n', pandas.Series(Y_train).value_counts())
print('Proportion of the target variable in the Training partition : \n', pandas.Series(Y_train).value_counts(normalize=True))
p_train = pandas.Series(Y_train).value_counts(normalize=True)
#print(p_train[0], p_train[1])


print('*************************Question 1 (b)*************************\n')
print('Count of the target variable in the Test partition : \n', pandas.Series(Y_test).value_counts())
print('Proportion of the target variable in the Test partition : \n', pandas.Series(Y_test).value_counts(normalize=True))
p_test = pandas.Series(Y_test).value_counts(normalize=True)
#print(p_test[0], p_test[1])

print('*************************Question 1 (c)*************************\n')
# What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?
probab_train = 0.7
probab_test = 0.3
probab_train_commercial = (p_train[1] * probab_train)/ (p_train[1] * probab_train + p_test[1] * probab_test)
print("The probability that an observation is in the Training partition given that CAR_USE = Commercial ::", probab_train_commercial)

print('*************************Question 1 (d)*************************\n')
 #What is the probability that an observation is in the Test partition given that CAR_USE = Private?
probab_test_private =( p_test[0] * probab_test) / (p_test[0] * probab_test  + p_train[0] * probab_train)

print("The probability that an observation is in the Test partition given that CAR_USE = Private::", probab_test_private)

print('***************************Question 2***************************\n')
########################################################################################

def EntropyOrdinalSplit(inData, split):
    dataTable = inData
    dataTable['LE_Split'] = (dataTable.iloc[:, 0] <= split)

    crossTable = pandas.crosstab(index=dataTable['LE_Split'], columns=dataTable.iloc[:, 1], margins=True, dropna=True)
    # print(crossTable)

    n_rows = crossTable.shape[0]
    n_columns = crossTable.shape[1]

    tableEntropy = 0 
    
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = crossTable.iloc[i_row, i_column] / crossTable.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * numpy.log2(proportion)
        #print('Row = ', i_row, 'Entropy =', row_entropy)
        #print(' ')
        tableEntropy += row_entropy * crossTable.iloc[i_row, (n_columns - 1)]
    tableEntropy = tableEntropy / crossTable.iloc[(n_rows - 1), (n_columns - 1)]

    return crossTable, tableEntropy


def minOrdinalEntropy(inData, setIntervals):
    minEntropy = sys.float_info.max
    minInterval = None
    minTable = None

    for i in range(setIntervals[0], setIntervals[len(setIntervals) - 1]):
        retTable, retEntropy = EntropyOrdinalSplit(inData=inData, split=i + 0.5)
        if retEntropy < minEntropy:
            minEntropy = retEntropy
            minInterval = i + 0.5
            minTable = retTable

    return minTable, minEntropy, minInterval


def EntropyNominalSplit(inData, subset):
    dataTable = inData
    dataTable['LE_Split'] = dataTable.iloc[:, 0].apply(lambda x: True if x in subset else False)

    crossTable = pandas.crosstab(index=dataTable['LE_Split'], columns=dataTable.iloc[:, 1], margins=True, dropna=True)
    # print(crossTable)

    n_rows = crossTable.shape[0]
    n_columns = crossTable.shape[1]

    tableEntropy = 0
    for i_row in range(n_rows - 1):
        row_entropy = 0
        for i_column in range(n_columns):
            proportion = crossTable.iloc[i_row, i_column] / crossTable.iloc[i_row, (n_columns - 1)]
            if proportion > 0:
                row_entropy -= proportion * numpy.log2(proportion)
        #print('Row = ', i_row, 'Entropy =', row_entropy)
        #print(' ')
        tableEntropy += row_entropy * crossTable.iloc[i_row, (n_columns - 1)]
    tableEntropy = tableEntropy / crossTable.iloc[(n_rows - 1), (n_columns - 1)]

    return crossTable, tableEntropy


def minNominalEntropy(inData, set):
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
        retTable, retEntropy = EntropyNominalSplit(inData=inData, subset=subset)
        if retEntropy < minEntropy:
            minEntropy = retEntropy
            minSubset1 = subset
            minSubset2 = subsetMap.get(subset)
            minTable = retTable

    return minTable, minEntropy, minSubset1, minSubset2

claim_history2 = claim_history[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()

claim_history2['EDUCATION'] = claim_history2['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})
#print(claim_history2)

claim_history2_train, claim_history2_test = train_test_split(claim_history2, test_size = 0.30, random_state=27513, stratify = claim_history['CAR_USE'])

# for layer 0 split
crossTable_edu, tableEntropy_edu, interval_edu = minOrdinalEntropy(inData=claim_history2_train[['EDUCATION', 'CAR_USE']],
                                                                         setIntervals=[0, 1, 2, 3, 4])
print(crossTable_edu, tableEntropy_edu, interval_edu)

crossTable_car_type, tableEntropy_car_type, subset1_car_type, subset2_car_type = minNominalEntropy(
    inData=claim_history2_train[['CAR_TYPE', 'CAR_USE']], set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(crossTable_car_type, tableEntropy_car_type, subset1_car_type, subset2_car_type)

crossTable_occu, tableEntropy_occu, subset1_occu, subset2_occu = minNominalEntropy(
    inData=claim_history2_train[['OCCUPATION', 'CAR_USE']],
    set=['Blue Collar', 'Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional', 'Student', 'Unknown'])
print(crossTable_occu, tableEntropy_occu, subset1_occu, subset2_occu)

# for layer 1 left node split
train_data_left_branch = claim_history2_train[claim_history2_train['OCCUPATION'].isin(subset1_occu)]

layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu = minOrdinalEntropy(
    inData=train_data_left_branch[['EDUCATION', 'CAR_USE']], setIntervals=[0, 1, 2, 3, 4])
print(layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu)

layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = minNominalEntropy(
    inData=train_data_left_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type)

layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu = minNominalEntropy(
    inData=train_data_left_branch[['OCCUPATION', 'CAR_USE']], set=subset1_occu)
print(layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu)

# for layer 1 right node split
train_data_right_branch = claim_history2_train[claim_history2_train['OCCUPATION'].isin(subset2_occu)]

layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu = minOrdinalEntropy(
    inData=train_data_right_branch[['EDUCATION', 'CAR_USE']], setIntervals=[0, 1, 2, 3, 4])
print(layer1_crossTable_edu, layer1_tableEntropy_edu, layer1_interval_edu)

layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type = minNominalEntropy(
    inData=train_data_right_branch[['CAR_TYPE', 'CAR_USE']],
    set=['Minivan', 'Panel Truck', 'Pickup', 'SUV', 'Sports Car', 'Van'])
print(layer1_crossTable_car_type, layer1_tableEntropy_car_type, layer1_subset1_car_type, layer1_subset2_car_type)

layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu = minNominalEntropy(
    inData=train_data_right_branch[['OCCUPATION', 'CAR_USE']], set=subset2_occu)
print(layer1_crossTable_occu, layer1_tableEntropy_occu, layer1_subset1_occu, layer1_subset2_occu)

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
print(claim_history2_train.shape[0])


print('*************************Question 2 (a)*************************\n')  
probab_commercial_train = claim_history2_train.groupby('CAR_USE').size()['Commercial'] / claim_history2_train.shape[0]
probab_private_train = claim_history2_train.groupby('CAR_USE').size()['Private'] / claim_history2_train.shape[0]

root_entropy = -((probab_commercial_train * math.log2(probab_commercial_train)) + (probab_private_train * math.log2(probab_private_train)))
print(f'root node entropy : {root_entropy}')

print('*************************Question 2 (b)*************************\n')

print(crossTable_edu, tableEntropy_edu, interval_edu)

print(crossTable_car_type, tableEntropy_car_type, subset1_car_type, subset2_car_type)

print(crossTable_occu, tableEntropy_occu, subset1_occu, subset2_occu)


print('*************************Question 2 (c)*************************\n')
print("the entropy of the split of the first layer", tableEntropy_occu)
print('*************************Question 2 (d)*************************\n')
print("There are four leaves")
print('*************************Question 2 (e)*************************\n')

print("Total count Commercial Private Class")
print("Leaf 1\n", total_count_l1, com_count_l1, pri_count_l1, class_l1)
print("Leaf 2\n", total_count_l2, com_count_l2, pri_count_l2, class_l2)
print("Leaf 3\n", total_count_l3, com_count_l3, pri_count_l3, class_l3)
print("Leaf 4\n", total_count_l4, com_count_l4, pri_count_l4, class_l4)


print('****************************Question 3***************************\n')

def predict_class(inData):
    if inData['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if inData['EDUCATION'] <= 0.5:
            return [0.6806378132118451, 0.3193621867881549]
        else:
            return [0.9133192389006343, 0.08668076109936575]
    else:
        if inData['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.006151953245155337, 0.9938480467548446]
        else:
            return [0.5464396284829721, 0.4535603715170279]


def predict_class_decision_tree(inData):
    out_data = numpy.ndarray(shape=(len(inData), 2), dtype=float)
    counter = 0
    for index, row in inData.iterrows():
        probability = predict_class(inData=row)
        out_data[counter] = probability
        counter += 1
    return out_data


claim_history3 = claim_history[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()

claim_history3_train, claim_history3_test = train_test_split(claim_history3, test_size = 0.30, random_state=27513)

## separate input and output variables in both part
train_data_x = claim_history3_train[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
train_data_y = claim_history3_train['CAR_USE']
test_data_x = claim_history3_test[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
test_data_y = claim_history3_test["CAR_USE"]

test_data_x['EDUCATION'] = test_data_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

print('*************************Question 3 (a)*************************\n')

# calculating threshold
threshold = claim_history3_train.groupby("CAR_USE").size()["Commercial"] / claim_history3_train.shape[0]

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

# calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(test_data_y, pred_y)
misclassification_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')

print('*************************Question 3 (b)*************************\n')


# calculating the root average squared error
RASE = 0.0
for y, ppy in zip(test_data_y, predProb_y):
    if y == 'Commercial':
        RASE += (1 - ppy) ** 2
    else:
        RASE += (0 - ppy) ** 2
RASE = numpy.sqrt(RASE / test_data_y.shape[0])
print(f'Root Average Squared Error: {RASE}')


print('*************************Question 3 (c)*************************\n')


y_true = 1.0 * numpy.isin(test_data_y, ['Commercial'])
AUC = metrics.roc_auc_score(y_true, predProb_y)
print(f'Area Under Curve: {AUC}')

print('*************************Question 3 (d)*************************\n')

# generating the coordinates for the roc curve
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(test_data_y, predProb_y, pos_label='Commercial')

# adding two dummy coordinates
one_minus_specificity = numpy.append([0], one_minus_specificity)
sensitivity = numpy.append([0], sensitivity)

one_minus_specificity = numpy.append(one_minus_specificity, [1])
sensitivity = numpy.append(sensitivity, [1])

# drawing the roc curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

