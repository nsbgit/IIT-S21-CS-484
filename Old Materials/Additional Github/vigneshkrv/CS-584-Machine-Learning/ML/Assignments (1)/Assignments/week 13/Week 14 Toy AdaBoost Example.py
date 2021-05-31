import graphviz
import numpy
import random
import sklearn.metrics as metrics
import sklearn.tree as tree

x_train = numpy.array([[0.1, 0.3],
                       [0.2, 0.2],
                       [0.3, 0.1],
                       [0.4, 0.4],
                       [0.5, 0.7],
                       [0.6, 0.5],
                       [0.7, 0.9],
                       [0.8, 0.8],
                       [0.8, 0.2],
                       [0.9, 0.8]], dtype = float)

y_train = numpy.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0], dtype = float)

w_train = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype = float)

# Build a classification tree on the training partition
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=60616)
treeFit = classTree.fit(x_train, y_train, w_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train, w_train)
print('Accuracy = ', accuracy)

# Update the weights
eventError = numpy.empty((10, 1))
predClass = numpy.empty((10, 1))

for i in range(10):
    if (y_train[i] == 0):
         eventError[i] = 0 - treePredProb[i,1]
    else:
         eventError[i] = 1 - treePredProb[i,1]

    if (treePredProb[i,1] >= treePredProb[i,0]):
         predClass[i] = 1
    else:
         predClass[i] = 0

    if (predClass[i] != y_train[i]):
         w_train[i] = 1 + numpy.abs(eventError[i])
    else:
         w_train[i] = numpy.abs(eventError[i])

print('Event Error:\n', eventError)
print('Predicted Class:\n', predClass)
print('Weight:\n', w_train)   

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60616)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
print('Accuracy = ', accuracy)

dot_data = tree.export_graphviz(treeFit,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['X1', 'X2'],
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
graph