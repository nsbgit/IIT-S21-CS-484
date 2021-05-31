# Load the PANDAS library
import pandas
hmeq = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',')

type(hmeq)

# Examine a portion of the data frame
print(hmeq)

# Inspect the BAD columns
print(hmeq.loc[:,'BAD'])

# Alternatively, point to the BAD column
print(hmeq['BAD'])

# Display the missing value indicators in the DEBTINC variable
pandas.isna(hmeq['DEBTINC'])

# Put the descriptive statistics into another dataframe
hmeq_descriptive = hmeq.describe()

# Horizontal frequency bar chart of BAD
hmeq.groupby('BAD').size().plot(kind='barh')

# Visualize the histogram of the DEBTINC variable
hmeq.hist(column='DEBTINC', bins=20)

# Visualize the histogram of the DELINQ variable
hmeq.hist(column='DELINQ', bins=20)

# Visualize the boxplot of the DEBTINC variable by BAD
hmeq.boxplot(column='DEBTINC', by='BAD', vert=False)

# Visualize the boxplot of the DELINQ variable by BAD
hmeq.boxplot(column='DELINQ', by='BAD', vert=False)

# Specify the target and the predictor variables
targetVar = 'BAD'
print(targetVar)

predList = ['DELINQ']
print(predList)

x_train = hmeq[predList]
y_train = hmeq[targetVar]

# How many missing values are there?
x_train.isnull().sum()
y_train.isnull().sum()

# Replace missing values in DEBTINC by -9000
# x_train['DEBTINC'] = x_train['DEBTINC'].fillna(-9000)

# Replace missing values in DELINQ by 0
x_train['DELINQ'] = x_train['DELINQ'].fillna(0)

# Load the TREE library from SKLEARN
from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)

hmeq_dt = classTree.fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(classTree.score(x_train, y_train)))

import graphviz
dot_data = tree.export_graphviz(hmeq_dt, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('hmeq_output')
