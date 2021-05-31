import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.tree as tree

trainData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\SpiralWithCluster.csv',
                            delimiter=',')

nObs = trainData.shape[0]

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

y_threshold = numpy.mean(trainData['SpectralCluster'])

x_train = trainData[['x','y']]
y_train = trainData['SpectralCluster']

# Suppose no limit on the maximum number of depths
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=60616)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
print('Accuracy = ', accuracy)

dot_data = tree.export_graphviz(treeFit,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['x', 'y'],
                                class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
graph

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['SpectralCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)

plt.plot([-4.5, 4.5], [-2.248, -2.248], color = 'black', linestyle = ':')
plt.plot([-4.5, 4.5], [2.121, 2.121], color = 'black', linestyle = ':')
plt.plot([-4.5, 4.5], [0.346, 0.346], color = 'black', linestyle = ':')
plt.plot([-1.874, -1.874], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([1.88, 1.88], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([3.042, 3.042], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([-3.131, -3.131], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([-4.5, 4.5], [-0.101, -0.101], color = 'black', linestyle = ':')
plt.plot([1.986, 1.986], [-3.5, 3.5], color = 'black', linestyle = ':')
plt.plot([-0.114, -0.114], [-3.5, 3.5], color = 'black', linestyle = ':')

plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'SpectralCluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()

# Build a classification tree on the training partition
w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(28)
ensemblePredProb = numpy.zeros((nObs, 2))

for iter in range(28):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=60616)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy[iter] = classTree.score(x_train, y_train, w_train)
    ensemblePredProb += accuracy[iter] * treePredProb
    
    # Update the weights
    eventError = numpy.empty((nObs, 1))
    predClass = numpy.empty((nObs, 1))

    for i in range(nObs):
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

# Calculate the final predicted probabilities
ensemblePredProb /= numpy.sum(accuracy)

trainData['predCluster'] = numpy.where(ensemblePredProb[:,1] >= 0.5, 1, 0)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['predCluster'] == i]
    plt.scatter(x = subData['x'],
                y = subData['y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Spiral with Cluster')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Cluster', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 15)
plt.show()
