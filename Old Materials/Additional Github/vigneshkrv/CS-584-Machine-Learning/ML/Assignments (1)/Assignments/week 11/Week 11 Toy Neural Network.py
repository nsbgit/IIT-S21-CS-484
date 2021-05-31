import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.neural_network as nn

# Create the toy data
x = numpy.zeros((201,1))
y = numpy.zeros((201,1))
for i in range(201):
   _x = (i - 100) / 100
   x[i,0] = _x
   if (_x < -0.7):
       y[i,0] = 1 + _x
   elif (_x < 0.2):
       y[i,0] = 0.3
   elif (_x < 0.8):
       y[i,0] = 0.2 + 0.5 * _x
   else:
       y[i,0] = 0.6

# Plot the toy data
plt.figure(figsize=(10,6))
plt.plot(x, y, linewidth = 0.5, marker = 'x')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPRegressor(hidden_layer_sizes = (nHiddenNeuron,)*nLayer, activation = 'tanh', verbose = True)
    # nnObj.out_activation_ = 'linear'
    fit_nn = nnObj.fit(x, y.ravel()) 
    pred_nn = nnObj.predict(x)

    print('Output Activiation Function:', nnObj.out_activation_)
    print('     R^2 of the Prediction.:', nnObj.score(x, y.ravel()))

    # Plot the prediction
    plt.figure(figsize=(10,6))
    plt.plot(x, y, linewidth = 2, marker = '+', color = 'black', label = 'Data')
    plt.plot(x, pred_nn, linewidth = 2, marker = 'o', color = 'red', label = 'Prediction')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
    plt.legend(fontsize = 12, markerscale = 3)
    plt.show()


Build_NN_Toy (nLayer = 1, nHiddenNeuron = 5)
Build_NN_Toy (nLayer = 1, nHiddenNeuron = 10)

Build_NN_Toy (nLayer = 2, nHiddenNeuron = 5)
Build_NN_Toy (nLayer = 2, nHiddenNeuron = 10)

Build_NN_Toy (nLayer = 3, nHiddenNeuron = 5)
Build_NN_Toy (nLayer = 3, nHiddenNeuron = 10)

Build_NN_Toy (nLayer = 5, nHiddenNeuron = 20)