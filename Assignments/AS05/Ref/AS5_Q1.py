import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.ensemble as ensemble
import sklearn.tree as tree
import sklearn.metrics as metrics

trainData = pandas.read_csv('WineQuality_Train.csv', delimiter=',')
testData = pandas.read_csv('WineQuality_Test.csv', delimiter=',')

nObs = trainData.shape[0]

x_train = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = trainData['quality_grp']
x_test = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = trainData['quality_grp']

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
treeFit = classTree.fit(x_train, y_train)
treePredProb = classTree.predict_proba(x_train)
accuracy = classTree.score(x_train, y_train)
print('Accuracy = ', accuracy)
print('Misclassification Rate Iteration 0 = ', 1-accuracy)

w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(50)
ensemblePredProb = numpy.zeros((nObs, 2))

for iter in range(50):
    classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20210415)
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    accuracy[iter] = classTree.score(x_train, y_train, w_train)
    ensemblePredProb += accuracy[iter] * treePredProb

    if (abs(1.0 - accuracy[iter]) < 0.0000001):
        break
    
    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - treePredProb[:,1]), (treePredProb[:,1]))
    predClass = numpy.where(treePredProb[:,1] >= 0.2, 1, 0)
    w_train = numpy.where(predClass != y_train, 2+numpy.abs(eventError), numpy.abs(eventError))

misclass=1-accuracy
misclass

