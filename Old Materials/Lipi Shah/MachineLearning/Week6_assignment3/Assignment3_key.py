import itertools
import math
import matplotlib.pyplot as plt
import numpy
import pandas

import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split

# Set some options for printing all the columns
pandas.set_option('display.max_columns', None)  
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', -1)
pandas.set_option('precision', 7)

CH = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\Claim_History.csv',
                     delimiter = ',', usecols = ['CAR_USE', 'CAR_TYPE', 'OCCUPATION', 'EDUCATION'])

# Print number of missing values per variable
print('Number of Missing Values:')
print(pandas.Series.sort_index(CH.isna().sum()))

nObs_all = CH.shape[0]

# Create a 70% Training partition and a 30% Test partition
CH_train, CH_test = train_test_split(CH, test_size = 0.3, random_state = 27513, stratify = CH['CAR_USE'])

nObs_train = CH_train.shape[0]
nObs_test = CH_test.shape[0]

prob_train = nObs_train / nObs_all
prob_test = nObs_test / nObs_all

print('Partition\t Count\t Poportion')
print('Training\t {:.0f} \t {:.6f}'.format(nObs_train, prob_train))
print('    Test\t {:.0f} \t {:.6f}'.format(nObs_test, prob_test))

# Question 1(a)
train_CU_freq = pandas.DataFrame(CH_train.groupby('CAR_USE').size(), columns = ['Count'])
train_CU_freq['Proportion'] = train_CU_freq / nObs_train

print('Training Partition Target Frequency Table')
print(train_CU_freq)
print('\n')

# Question 1(b)
test_CU_freq = pandas.DataFrame(CH_test.groupby('CAR_USE').size(), columns = ['Count'])
test_CU_freq['Proportion'] = test_CU_freq / nObs_test

print('Test Partition Target Frequency Table')
print(test_CU_freq)
print('\n')

# Question 1(c)(d)
# Calculate the Prob(CAR_USE | Partition) * Prob(Partition)
pCU_train = train_CU_freq['Proportion'] * prob_train
pCU_test = test_CU_freq['Proportion'] * prob_test

# Calculate the Prob(CAR_USE | Training) * Prob(Training) + Prob(CAR_USE | Test) * Prob(Test)
pDenom = pCU_train + pCU_test

pTrain_CU = pCU_train / pDenom
pTest_CU = pCU_test / pDenom

# Question 1(c)
print('Conditional Probability that Partition = Train Given CAR_USE')
print(pTrain_CU)
print('\n')

# Question 1(d)
print('Conditional Probability that Partition = Test Given CAR_USE')
print(pTest_CU)
print('\n')

# Question 2
# Define a function to compute the entropy metric of a split
def EntropyNominalSplit (
        inData,          # input data frame (predictor in column 0 and target in column 1)
        split,           # split set
        debug = 'N'):    # debug flag (Y/N)
    
    countTable = pandas.crosstab(index = (inData.iloc[:,0]).isin(split),
                                 columns = inData.iloc[:,1],
                                 margins = False, dropna = True)
    fractionTable = countTable.div(countTable.sum(1), axis = 'index')

    if (debug == 'Y'):
        print('Criterion for Being True: \n', split)
        print()
        print('countTable: \n', countTable)
        print()
        print('fractionTable: \n', fractionTable)
        print()

    nRows = fractionTable.shape[0]
    nColumns = fractionTable.shape[1]

    tableEntropy = 0.0
    tableN = 0
    for iRow in range(nRows):
        rowEntropy = 0.0
        rowN = 0
        for iColumn in range(nColumns):
            rowN += countTable.iloc[iRow, iColumn]
            proportion = fractionTable.iloc[iRow, iColumn]
            if (proportion > 0):
                rowEntropy -= (proportion * math.log2(proportion))

        if (debug == 'Y'):
           print('Row N = ', rowN, ' Row Entropy = ', rowEntropy)

        tableEntropy += (rowN * rowEntropy)
        tableN += rowN
    tableEntropy = tableEntropy /  tableN

    if (debug == 'Y'):
        print('Table N = ', tableN, ' Table Entropy = ', tableEntropy)

    return(tableEntropy)

def GetNominalSplit (
        inData,          # input data frame (predictor in column 0 and target in column 1)
        debug = 'N'):    # debug flag (Y/N)

    catPred = set(inData.iloc[:,0])
    nCatPred = len(catPred)

    if (debug == 'Y'):
       print('Predictor: ', inData.columns[0])
       print('Number of Categories =', nCatPred)
       print('Categories: \n', sorted(catPred))
       print('Number of Possible Splits = ', (2**(nCatPred-1) - 1))

    treeResult = pandas.DataFrame(columns = ['# Left', 'Left Branch', 'Right Branch', 'Entropy'])
    for i in range(1, round((nCatPred+1)/2)):
        allComb_i = itertools.combinations(catPred, i)
        for comb in list(allComb_i):
            combComp = catPred.difference(comb)
            EV = EntropyNominalSplit(inData, comb)
            treeResult = treeResult.append(pandas.DataFrame([[i, sorted(comb), sorted(combComp), EV]], 
                                           columns = ['# Left', 'Left Branch', 'Right Branch', 'Entropy']),
                                           ignore_index = True)

    treeResult = treeResult.sort_values(by = 'Entropy', axis = 0, ascending = True)
    return(treeResult)

def GetOrdinalSplit (
        inData,          # input data frame (predictor in column 0 and target in column 1)
        predValue,       # predictor values in ascending order
        debug = 'N'):    # debug flag (Y/N)

    catPred = set(predValue)
    nCatPred = len(catPred)

    if (debug == 'Y'):
        print('Predictor: ', inData.columns[0])
        print('Number of Categories =', nCatPred)
        print('Categories: \n', sorted(catPred))
        print('Number of Possible Splits = ', (nCatPred-1))

    treeResult = pandas.DataFrame(columns = ['# Left', 'Left Branch', 'Right Branch', 'Entropy'])
    for i in range(1, nCatPred):
        comb = list(predValue[0:i])
        combComp = list(predValue[i:nCatPred])
        print(comb)
        print(combComp)
        EV = EntropyNominalSplit(inData, comb)
        treeResult = treeResult.append(pandas.DataFrame([[i, comb, combComp, EV]], 
                                       columns = ['# Left', 'Left Branch', 'Right Branch', 'Entropy']),
                                       ignore_index = True)

    treeResult = treeResult.sort_values(by = 'Entropy', axis = 0, ascending = True)
    return(treeResult)

# Question 2(a)
inData = CH_train[['CAR_TYPE', 'CAR_USE']]
entropy_Root = EntropyNominalSplit(inData, [], 'Y')
print('Root Node Entropy = ', entropy_Root)

# Question 2(b)
inData = CH_train[['CAR_TYPE', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData, 'Y')

inData = CH_train[['OCCUPATION', 'CAR_USE']]
tree_OCCUPATION = GetNominalSplit(inData, 'Y')

inData = CH_train[['EDUCATION', 'CAR_USE']]
tree_EDUCATION = GetOrdinalSplit(inData, ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors'], 'Y')

# Question 2(c)
# See the counts of the optional split
inData = CH_train[['OCCUPATION', 'CAR_USE']]
split = list(['Blue Collar', 'Student', 'Unknown'])
EntropyNominalSplit (inData, split, 'Y')

# Question 2(d)
leftBranch = CH_train[CH_train['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]

inData = leftBranch[['CAR_TYPE', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData, 'Y')

inData = leftBranch[['OCCUPATION', 'CAR_USE']]
tree_OCCUPATION = GetNominalSplit(inData, 'Y')

inData = leftBranch[['EDUCATION', 'CAR_USE']]
tree_EDUCATION = GetOrdinalSplit(inData, ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors'], 'Y')

rightBranch = CH_train[~CH_train['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]

inData = rightBranch[['CAR_TYPE', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData, 'Y')

inData = rightBranch[['OCCUPATION', 'CAR_USE']]
tree_OCCUPATION = GetNominalSplit(inData, 'Y')

inData = rightBranch[['EDUCATION', 'CAR_USE']]
tree_EDUCATION = GetOrdinalSplit(inData, ['Below High School', 'High School', 'Bachelors', 'Masters', 'Doctors'], 'Y')

# Question 2(e)
# Define the four leaves
def set_leaf (row):
    if (numpy.isin(row['OCCUPATION'], ['Blue Collar', 'Student', 'Unknown'])):
        if (numpy.isin(row['EDUCATION'], ['Below High School'])):
            Leaf = 0
        else:
            Leaf = 1
    else:
        if (numpy.isin(row['CAR_TYPE'], ['Minivan', 'SUV', 'Sports Car'])):
            Leaf = 2
        else:
            Leaf = 3

    return(Leaf)

CH_train = CH_train.assign(Leaf = CH_train.apply(set_leaf, axis = 1))

countTable = pandas.crosstab(index = CH_train['Leaf'], columns = CH_train['CAR_USE'],
                             margins = False, dropna = True)
predProbCAR_USE = countTable.div(countTable.sum(1), axis = 'index')

def leaf_prob_Commercial (row):
    predProb = predProbCAR_USE.iloc[row['Leaf']]
    pCAR_USE_Commercial = predProb['Commercial']
    return(pCAR_USE_Commercial)

def leaf_prob_Private (row):
    predProb = predProbCAR_USE.iloc[row['Leaf']]
    pCAR_USE_Private = predProb['Private']
    return(pCAR_USE_Private)

print('countTable:')
print(countTable)
print('\n')
print('predProbCAR_USE:')
print(predProbCAR_USE)
print('\n')

# Question 3
CH_test = CH_test.assign(Leaf = CH_test.apply(set_leaf, axis = 1))
CH_test = CH_test.assign(pCAR_USE_Commercial = CH_test.apply(leaf_prob_Commercial, axis = 1))
CH_test = CH_test.assign(pCAR_USE_Private = CH_test.apply(leaf_prob_Private, axis = 1))

threshold = train_CU_freq.loc['Commercial', 'Proportion']
CH_test['pred_CAR_USE'] = numpy.where(CH_test['pCAR_USE_Commercial'] >= threshold, 'Commercial', 'Private')

# Question 3(a)
print('Threshold = ', threshold)

confuse_Matrix = metrics.confusion_matrix(CH_test['CAR_USE'], CH_test['pred_CAR_USE'])
print('Confusion Matrix')
print(confuse_Matrix)
print('\n')

misClassRate = 1.0 - metrics.accuracy_score(CH_test['CAR_USE'], CH_test['pred_CAR_USE'])
print('Misclassification Rate') 
print(misClassRate)
print('\n')

# Question 3(b)
CH_test['ASE'] = numpy.where(CH_test['CAR_USE'] == 'Commercial', (1.0 - CH_test['pCAR_USE_Commercial'])**2, (0 - CH_test['pCAR_USE_Commercial'])**2)
RASE = numpy.sqrt(numpy.sum(CH_test['ASE'] / CH_test.shape[0]))
print('Root Average Squared Error')
print(RASE)
print('\n')

# Question 3(c)
Y_true = 1.0 * numpy.isin(CH_test['CAR_USE'], ['Commercial'])
AUC = metrics.roc_auc_score(Y_true, CH_test['pCAR_USE_Commercial'])
print('Area Under Curve')
print(AUC)
print('\n')

# Question 3(d)
# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(CH_test['CAR_USE'], CH_test['pCAR_USE_Commercial'], pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()

