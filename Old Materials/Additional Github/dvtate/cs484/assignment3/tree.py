# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.tree as tree

# Read data
df = pandas.read_csv('./claim_history.csv')

'''
Target: CAR_USE
    The usage of car.  This field has two categories, namely, Commercial and Private.  The Commercial category is the Event value.

Nominal:
- CAR_TYPE: Model-Type of car
- OCCUPATION: Car owner job

Ordinal:
- EDUCATION: education level of car owner
'''

# Partition data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.30, random_state=60616)

def freq_analysis(df):
    # Show frequencies
    n_priv = len(df[df['CAR_USE'] == 'Private'])
    n_comm = len(df[df['CAR_USE'] == 'Commercial'])
    print('Private: %s (%s%%)' % (n_priv, 100 * n_priv / (n_priv + n_comm)))
    print('Commercial: %s (%s%%)' % (n_comm, 100 * n_comm / (n_priv + n_comm)))

    # Frequency chart
    df.groupby('CAR_USE').size().plot(kind='barh')
    plt.title("Car use frequency")
    plt.xlabel("Number of usages")
    plt.ylabel("Car Usage")
    plt.grid(axis="x")
    plt.show()

# Do analysis
print('train data:')
freq_analysis(train_data)
print('\ntest data:')
freq_analysis(test_data)


# Make tree
ctree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)

# Train
one_hot_inputs = pandas.get_dummies(
    train_data[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']],
    drop_first = True)
targets = train_data[['CAR_USE']]
dt = ctree.fit(one_hot_inputs, targets)

# Make graph
import graphviz
dot_data = tree.export_graphviz(
    dt,
    out_file = None,
    impurity = True,
    filled = True,
    # feature_names = ['Car Type', 'Driver Occupation', 'Education Level'],
    )
graph = graphviz.Source(dot_data)
print(graph)
graph.render('graph')

# Print column names that were assigned by pandas.get_dummies
print('Some labels:')
for i, col in enumerate(one_hot_inputs):
    print('X[%s] = %s' % (i, col))
    if i > 10:
        break


print('\n')

# Find misclassification rate for test data
one_hot_inputs = pandas.get_dummies(
    test_data[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']],
    drop_first = True)
targets = test_data[['CAR_USE']]
print('misclassification rate: %s%%' % (100 - 100 * dt.score(one_hot_inputs, targets)))

from sklearn.metrics import mean_squared_error
# print('root mean squared error: ', mean_squared_error(
#     dt.predict(one_hot_inputs), targets))