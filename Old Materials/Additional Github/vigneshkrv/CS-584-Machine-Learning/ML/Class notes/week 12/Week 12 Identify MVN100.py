import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neighbors as kNN
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.tree as tree
import statsmodels.api as sm

trainData = pandas.read_excel('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\MVN100.xlsx',
                              sheet_name = 'MVN100', usecols = ['Group', 'X', 'Y'])

y_threshold = trainData['Group'].mean()

# Scatterplot that uses prior information of the grouping variable
carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['Group'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.title('Prior Group Information')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Group', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try the various number of nearest neighbor classifier
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']
kNN_accuracy = numpy.zeros((20,2))
for k in numpy.arange(1,21,1):
    neigh = kNN.KNeighborsClassifier(n_neighbors=k, algorithm = 'brute', metric = 'euclidean')
    nbrs = neigh.fit(xTrain, yTrain)

    # See the classification result
    kNN_accuracy[k-1,0] = k
    kNN_accuracy[k-1,1] = neigh.score(xTrain, yTrain)

print('Nearest Neighbor Accuracy')
print(kNN_accuracy)

plt.plot(numpy.arange(1,21,1), kNN_accuracy, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(numpy.arange(1, 11, step = 1))
plt.show()

neigh = kNN.KNeighborsClassifier(n_neighbors=2, algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(xTrain, yTrain)
trainData['_PredictedClass_'] = nbrs.predict(xTrain)
kNN_Mean = trainData.groupby('_PredictedClass_').mean()

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.scatter(x = kNN_Mean['X'], y = kNN_Mean['Y'], c = 'black', marker = 'X', s = 100)
plt.grid(True)
plt.title('2-Nearest Neighbors')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try the decision tree classificer
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']
tree_accuracy = numpy.zeros((10,2))
for k in numpy.arange(1,11,1):
    objTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=k, random_state=20191106)
    thisFit = objTree.fit(xTrain, yTrain)
    tree_accuracy[k-1,0] = k
    tree_accuracy[k-1,1] = objTree.score(xTrain, yTrain)


objTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=20191106)
thisFit = objTree.fit(xTrain, yTrain)

dot_data = tree.export_graphviz(thisFit,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['X', 'Y'],
                                class_names = ['0', '1'])
graph = graphviz.Source(dot_data)
graph

y_predProb = thisFit.predict_proba(xTrain)
trainData['_PredictedClass_'] = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)

tree_Mean = trainData.groupby('_PredictedClass_').mean()
print(tree_Mean)

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.scatter(x = tree_Mean['X'], y = tree_Mean['Y'], c = 'black', marker = 'X', s = 100)
plt.plot([-3,3], [0.342,0.342], linestyle = ':', linewidth = 3)
plt.plot([0.823, 0.823], [-3,3], linestyle = ':', linewidth = 3)
plt.grid(True)
plt.title('Classification Tree, Depth = 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try the logistic classifier
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']
xTrain = sm.add_constant(xTrain, prepend=True)
logit = sm.MNLogit(yTrain, xTrain)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='bfgs', full_output = True, maxiter = 100, tol = 1e-8, retall = True)
print(thisFit.summary())

y_predProb = thisFit.predict(xTrain)
trainData['_PredictedClass_'] = numpy.where(y_predProb[1] >= y_threshold, 1, 0)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, trainData['_PredictedClass_'].values))

logistic_Mean = trainData.groupby('_PredictedClass_').mean()
print(logistic_Mean)

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.scatter(x = logistic_Mean['X'], y = logistic_Mean['Y'], c = 'black', marker = 'X', s = 100)
plt.grid(True)
plt.title('Multinomial Logistic')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Try MLP Neural Network
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']

result = pandas.DataFrame(columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Accuracy'])
for nLayer in numpy.arange(1,5):
    for nHiddenNeuron in numpy.arange(1,11,1):
        nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                                 activation = 'relu', verbose = False,
                                 solver = 'lbfgs', learning_rate_init = 0.1,
                                 max_iter = 5000, random_state = 20191106)
        # nnObj.out_activation_ = 'identity'
        thisFit = nnObj.fit(xTrain, yTrain) 
        y_predProb = nnObj.predict_proba(xTrain)
        y_pred = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)

        mlp_loss = nnObj.loss_
        mlp_accuracy = metrics.accuracy_score(yTrain, y_pred)
        result = result.append(pandas.DataFrame([[nLayer, nHiddenNeuron, mlp_loss, mlp_accuracy]], 
                               columns = ['nLayer', 'nHiddenNeuron', 'Loss', 'Accuracy']))

xTrain = trainData[['X','Y']]
yTrain = trainData['Group']

nLayer = 3
nHiddenNeuron = 7
nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                         activation = 'relu', verbose = False,
                         solver = 'lbfgs', learning_rate_init = 0.1,
                         max_iter = 5000, random_state = 20191106)
thisFit = nnObj.fit(xTrain, yTrain) 
y_predProb = nnObj.predict_proba(xTrain)
trainData['_PredictedClass_'] = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)

mlp_Mean = trainData.groupby('_PredictedClass_').mean()
print(mlp_Mean)

carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.scatter(x = mlp_Mean['X'], y = mlp_Mean['Y'], c = 'black', marker = 'X', s = 100)
plt.grid(True)
plt.title('MLP (3 Layers, 7 Neurons)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

# Build Support Vector Machine classifier
xTrain = trainData[['X','Y']]
yTrain = trainData['Group']

svm_Model = svm.SVC(kernel = 'linear', random_state = 20191106, max_iter = -1)
thisFit = svm_Model.fit(xTrain, yTrain)
y_predictClass = thisFit.predict(xTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)

print('Intercept = ', thisFit.intercept_)
print('Coefficients = ', thisFit.coef_)

# get the separating hyperplane
w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = numpy.linspace(-3, 3)
yy = a * xx - (thisFit.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = thisFit.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = thisFit.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

cc = thisFit.support_vectors_

# plot the line, the points, and the nearest vectors to the plane
carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['X'],
                y = subData['Y'], c = carray[i], label = i, s = 25)
plt.scatter(x = svm_Mean['X'], y = svm_Mean['Y'], c = 'black', marker = 'X', s = 100)
plt.plot(xx, yy, color = 'black', linestyle = '-')
plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
plt.scatter(cc[:,0], cc[:,1], color = 'black', marker = '+', s = 100)
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

