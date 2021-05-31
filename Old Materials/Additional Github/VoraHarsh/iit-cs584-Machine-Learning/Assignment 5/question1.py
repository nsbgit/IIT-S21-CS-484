import pandas as pd
import numpy as np
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

spiral_df = pd.read_csv('SpiralWithCluster.csv', delimiter=',', usecols=['x', 'y', 'SpectralCluster'])

print("Q1.a)(5 points) What percent of the observations have SpectralCluster equals to 1?")
total_obs = spiral_df.shape[0]
no_obs_1 = spiral_df[spiral_df['SpectralCluster'] == 1].shape[0]
per_obs_1 = (no_obs_1) / total_obs
print((per_obs_1*100), "percent of the observation have SpectralCluster equals to 1")

threshold_y = per_obs_1
print("")

print("Q1.b)(15 points) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.  You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1. Your search will be done over a grid that is formed by cross-combining the following attributes: (1) activation function: identity, logistic, relu, and tanh; (2) number of hidden layers: 1, 2, 3, 4, and 5; and (3) number of neurons: 1 to 10 by 1.  List your optimal neural network for each activation function in a table.  Your table will have four rows, one for each activation function.  Your table will have six columns: (1) activation function, (2) number of layers, (3) number of neurons per layer, (4) number of iterations performed, (5) the loss value, and (6) the misclassification rate.")

x_Train = spiral_df[['x', 'y']]
y_Train = spiral_df['SpectralCluster']

def mlpClassifier(actFunction,a):
    result = pd.DataFrame(columns=['Index','ActivationFunction', 'nLayers', 'nNeuronsPerLayer', 'nIterations', 'Loss', 'MisclassificationRate'])
    for nLayer in np.arange(1, 6):
        for npl in np.arange(1, 11, 1):
            nn_Obj = nn.MLPClassifier(hidden_layer_sizes=(npl,) * nLayer, activation=actFunction, verbose=False,solver='lbfgs', learning_rate_init=0.1, max_iter=5000, random_state=20200408)
            thisFit = nn_Obj.fit(x_Train, y_Train)
            y_predProb = nn_Obj.predict_proba(x_Train)
            y_pred = np.where(y_predProb[:, 1] >= threshold_y, 1, 0)
            activation_fn = nn_Obj.activation
            mlp_loss = nn_Obj.loss_
            niter = nn_Obj.n_iter_
            activationOut = nn_Obj.out_activation_
            misclassification = 1 - (metrics.accuracy_score(y_Train, y_pred))
            result = result.append(pd.DataFrame(
                [[a,activation_fn, nLayer, npl, niter, mlp_loss, misclassification, activationOut]],
                columns=['Index','ActivationFunction', 'nLayers', 'nNeuronsPerLayer', 'nIterations', 'Loss', 'MisclassificationRate', 'ActivationFunction_output']))
    return (result[result.Loss == result.Loss.min()]),activationOut

act1, output_activation = mlpClassifier('identity',0)
act2, output_activation = mlpClassifier('logistic',1)
act3, output_activation = mlpClassifier('relu',2)
act4, output_activation = mlpClassifier('tanh',3)

table = pd.concat([act1, act2, act3, act4]).set_index("Index")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(table)
print("")

print("Q1.c)(5 points) What is the activation function for the output layer?")

print("Activation function of output layer is",output_activation)
print("")

print("Q1.d)(5 points) Which activation function, number of layers, and number of neurons per layer give the lowest loss and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are performed?")

print(table[table.Loss == table.Loss.min()])
print("")

print("Q1.e)(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.")

x_Train = spiral_df[['x', 'y']]
y_Train = spiral_df['SpectralCluster']

nLayer = 4
nHiddenNeuron = 10
nn_Obj = nn.MLPClassifier(hidden_layer_sizes=(nHiddenNeuron,) * nLayer,
                          activation='relu', verbose=False,
                          solver='lbfgs', learning_rate_init=0.1,
                          max_iter=5000, random_state=20200408)
thisFit = nn_Obj.fit(x_Train, y_Train)
y_predProb = nn_Obj.predict_proba(x_Train)
spiral_df['_PredictedClass_'] = np.where(y_predProb[:, 1] >= threshold_y, 1, 0)

mlp_Mean = spiral_df.groupby('_PredictedClass_').mean()
print(mlp_Mean)

carray = ['red', 'blue']
plt.figure(figsize=(10, 10))
for i in range(2):
    subData = spiral_df[spiral_df['_PredictedClass_'] == i]
    plt.scatter(x=subData['x'], y=subData['y'], c=carray[i], label=i, s=25)
plt.grid(True)
plt.title('MLP (relu, 4 Layers, 10 Neurons)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title='Predicted Class', loc='best', bbox_to_anchor=(1, 1), fontsize=14)
plt.show()
print("")

print("Q1.f)(5 points) What is the count, the mean and the standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to the 10 decimal places.")
spiral_df['y_pred_1'] = y_predProb[:, 1]
pd.set_option('float_format', '{:.10f}'.format)
print(spiral_df[spiral_df['_PredictedClass_'] == 1]['y_pred_1'].describe()[['count','mean','std']])