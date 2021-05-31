import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.tree

'''
Please use the following information for answering Questions 11, 12, and 13.
We are interested in studying the effects of the Vehicle Age on the Claim Indicator. We particularly
want to optimally separate the Vehicle Age into two groups.
You are given a two-way table and are asked to build a decision tree model using the Entropy criterion
to discover the groups. We will treat the Vehicle Age as an ordinal predictor and the Claim Indicator as
a nominal target variable. The order of the Vehicle Age is ‘1 to 3’ < ‘4 to 7’ < ‘8 to 10’ < ’11 and Above’.
'''

# Make data
# 1-3 | 4-7 | 8-10 | 11+ | Claim
data = []
data += [[1, 0, 0, 0, 0]] * 1731
data += [[1, 0, 0, 0, 1]] * 846
data += [[0, 1, 0, 0, 0]] * 1246
data += [[0, 1, 0, 0, 1]] * 490
data += [[0, 0, 1, 0, 0]] * 1412
data += [[0, 0, 1, 0, 1]] * 543
data += [[0, 0, 0, 1, 0]] * 2700
data += [[0, 0, 0, 1, 1]] * 690
df = pandas.DataFrame(
    data = data,
    columns = ['Age 1-3', 'Age 4-7', 'Age 8-10', 'Age 11+', 'Claim'])

# Make decision tree
ctree = sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
X = df[['Age 1-3', 'Age 4-7', 'Age 8-10', 'Age 11+']]
Y = df['Claim']
dt = ctree.fit(X, Y)

# Make graph
import graphviz
dot_data = sklearn.tree.export_graphviz(
    dt,
    out_file = None,
    impurity = True,
    filled = True,
    feature_names = ['Age 1-3', 'Age 4-7', 'Age 8-10', 'Age 11+'],
    )
graph = graphviz.Source(dot_data)
print(graph)
graph.render('graph')
