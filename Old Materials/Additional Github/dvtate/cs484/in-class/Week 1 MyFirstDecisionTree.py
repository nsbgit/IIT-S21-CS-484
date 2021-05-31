# Load the necessary libraries
import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.tree as tree

# load data
trainData = pandas.read_csv('hmeq.csv', delimiter=',', usecols = ['BAD', 'DELINQ'])

# Remove all missing observations
trainData = trainData.dropna()

# Examine a portion of the data frame
print(trainData)

# Put the descriptive statistics into another dataframe
trainData_descriptive = trainData.describe()
print(trainData_descriptive)

# Horizontal frequency bar chart of BAD
trainData.groupby('BAD').size().plot(kind='barh')
plt.title("Barchart of BAD")
plt.xlabel("Number of Observations")
plt.ylabel("BAD")
plt.grid(axis="x")
plt.show()

# Visualize the histogram of the DELINQ variable
trainData.hist(column='DELINQ', bins=15)
plt.title("Histogram of DELINQ")
plt.xlabel("DELINQ")
plt.ylabel("Number of Observations")
plt.xticks(numpy.arange(0, 15, step=1))
plt.grid(axis="x")
plt.show()

# Visualize the boxplot of the DELINQ variable by BAD
trainData.boxplot(column='DELINQ', by='BAD', vert=False)
plt.title("Boxplot of DELINQ by Levels of BAD")
plt.suptitle("")
plt.xlabel("DELINQ")
plt.ylabel("BAD")
plt.grid(axis="y")
plt.show()

X_inputs = trainData[['DELINQ']]
Y_target = trainData[['BAD']]


trainData = pandas.DataFrame({'A': [1,2,3,4,5],
                            'B': [262, 1007, 1662, 1510, 559]})
X_inputs = trainData[['B']]
Y_target = trainData[['A']]

# Load the TREE library from SKLEARN
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)

hmeq_dt = classTree.fit(X_inputs, Y_target)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(classTree.score(X_inputs, Y_target)))

dot_data = tree.export_graphviz(hmeq_dt,
                                out_file=None,
                                impurity = True, filled = True)
print(dot_data)

graph = graphviz.Source(dot_data)

print(graph, type(graph))

graph.render('test')