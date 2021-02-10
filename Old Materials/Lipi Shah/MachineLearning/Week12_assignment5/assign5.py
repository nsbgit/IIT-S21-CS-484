
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.neural_network as nn
import sklearn.svm as svm
import statsmodels.api as sm

trainData = pandas.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week12_assignment5\\SpiralWithCluster.csv')

y_threshold = trainData['SpectralCluster'].mean()

#########################################Question 1 a ########################################
scCounts = trainData['SpectralCluster'].value_counts()
percentObv = (scCounts[1] / (scCounts[0]+scCounts[1]))*100
print("The percentage of the observations have SpectralCluster equals to 1 is", percentObv)

#########################################Question 1 b ########################################

# MLP Neural Network
xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']

def mlpClassifier(actFunction):
    lossList = []
    result = pandas.DataFrame(columns = ['activation','niter','nLayer', 'nHiddenNeuron', 'Loss', 'misclassification'])
    for nLayer in numpy.arange(1,5):
        for nHiddenNeuron in numpy.arange(1,11,1):
            nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                                     activation = actFunction, verbose = False,
                                     solver = 'lbfgs', learning_rate_init = 0.1,
                                     max_iter = 5000, random_state = 20191108)
            #nnObj.out_activation_ = 'identity'
            thisFit = nnObj.fit(xTrain, yTrain) 
            y_predProb = nnObj.predict_proba(xTrain)
            y_pred = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)
            activationfn = nnObj.activation
            mlp_loss = nnObj.loss_
            #lossList.append(mlp_loss)
            niter = nnObj.n_iter_
            activationOut = nnObj.out_activation_
            misclassification = 1 - (metrics.accuracy_score(yTrain, y_pred))
            result = result.append(pandas.DataFrame([[activationfn, niter, nLayer, nHiddenNeuron,mlp_loss, misclassification,activationOut]], 
                                   columns = ['activation','niter', 'nLayer', 'nHiddenNeuron', 'Loss', 'misclassification', 'output layer activation function']))
            #print(result) 
    return(result[result.Loss == result.Loss.min()])


act1 = mlpClassifier('identity')
act2 = mlpClassifier('logistic')
act3 = mlpClassifier('tanh')
act4 = mlpClassifier('relu')

table = pandas.concat([act1,act2,act3,act4]).reset_index()
#table.drop(columns=['index'], axis = 1)
print(table)

#########################################Question 1 c ########################################

print("Activation function of outlayer is Logistic")

#########################################Question 1 d ########################################

print(table[table.Loss == table.Loss.min()])


#########################################Question 1 e ########################################

xTrain = trainData[['x','y']]
yTrain = trainData['SpectralCluster']
 
nLayer = 4
nHiddenNeuron = 8
nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                         activation = 'relu', verbose = False,
                         solver = 'lbfgs', learning_rate_init = 0.1,
                         max_iter = 5000, random_state = 20191108)
thisFit = nnObj.fit(xTrain, yTrain) 
y_predProb = nnObj.predict_proba(xTrain)
trainData['_PredictedClass_'] = numpy.where(y_predProb[:,1] >= y_threshold, 1, 0)


mlp_Mean = trainData.groupby('_PredictedClass_').mean()
#mlp_count = trainData.groupby('_PredictedClass_').count()
#mlp_std = trainData.groupby('_PredictedClass_').std()
print(mlp_Mean)
#print(mlp_count)
#print(mlp_std)

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['x'],                 y = subData['y'], c = carray[i], label = i, s = 25)
###############################need to check x y##################
plt.scatter(x = mlp_Mean['x'], y = mlp_Mean['y'], c = 'black', marker = 'X', s = 100)
###############################need to check x y##################
plt.grid(True)
plt.title('MLP (3 Layers, 7 Neurons)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()

#########################################Question 1 f ########################################
countPP = sum(y_predProb[:,1])
meanPP = numpy.mean(y_predProb[:,1])
stdPP = numpy.std(y_predProb[:,1])
print(countPP)
print(meanPP)
print(stdPP)




# 
## Build Support Vector Machine classifier
#xTrain = trainData[['x','y']]
#yTrain = trainData['SpectralCluster']
#
#svm_Model = svm.SVC(kernel = 'linear', random_state = 20191108, max_iter = -1, decision_function_shape  = 'ovr')
#thisFit = svm_Model.fit(xTrain, yTrain)
#y_predictClass = thisFit.predict(xTrain)
#
#print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
#trainData['_PredictedClass_'] = y_predictClass
#
#svm_Mean = trainData.groupby('_PredictedClass_').mean()
#print(svm_Mean)
#
#print('Intercept = ', thisFit.intercept_)
#print('Coefficients = ', thisFit.coef_)
#
## get the separating hyperplane
#w = thisFit.coef_[0]
#a = -w[0] / w[1]
#xx = numpy.linspace(-3, 3)
#yy = a * xx - (thisFit.intercept_[0]) / w[1]
#
## plot the parallels to the separating hyperplane that pass through the
## support vectors
#b = thisFit.support_vectors_[0]
#yy_down = a * xx + (b[1] - a * b[0])
#
#b = thisFit.support_vectors_[-1]
#yy_up = a * xx + (b[1] - a * b[0])
#
## plot the line, the points, and the nearest vectors to the plane
#carray = ['red', 'green']
#plt.figure(figsize=(10,10))
#for i in range(2):
#    subData = trainData[trainData['_PredictedClass_'] == i]
#    plt.scatter(x = subData['x'],
#                y = subData['y'], c = carray[i], label = i, s = 25)
#plt.scatter(x = svm_Mean['x'], y = svm_Mean['y'], c = 'black', marker = 'x', s = 100)
#plt.plot(xx, yy, color = 'black', linestyle = '-')
#plt.plot(xx, yy_down, color = 'blue', linestyle = '--')
#plt.plot(xx, yy_up, color = 'blue', linestyle = '--')
#plt.scatter(cc[:,0], cc[:,1], color = 'black', marker = '+', s = 100)
#plt.grid(True)
#plt.title('Support Vector Machines')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
#plt.show()