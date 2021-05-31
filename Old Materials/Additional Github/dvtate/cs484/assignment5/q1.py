
import numpy
import pandas
import sklearn
import sklearn.svm
import sklearn.neural_network
import matplotlib.pyplot as plt


df = pandas.read_csv('SpiralWithCluster.csv')
X = df[['x', 'y']]
Y = df['SpectralCluster']

# A. Get frequency
total = 0
cluster1 = 0
for c in df['SpectralCluster']:
    if c == 1 or c == '1':
        cluster1 += 1
    total += 1

threshold = cluster1 / total
print('Opservations with SpectralCluster=1 : %s%%' % (100 * threshold)) # 50%


'''
b. (20 points) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.
    You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1.
    Your search will be done over a grid that is formed by cross-combining the following attributes:
        (1) activation function: identity, logistic, relu, and tanh;
        (2) number of hidden layers: 1, 2, 3, 4, and 5; and
        (3) number of neurons: 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10.
    List your optimal neural network for each activation function in a table.
    Your table will have four rows, one for each activation function.
    Your table will have six columns:
        (1) activation function,
        (2) number of layers,
        (3) number of neurons per layer,
        (4) number of iterations performed,
        (5) the loss value, and
        (6) the misclassification rate.
'''

# Util to make nn classifier
def Build_NN_Class (actFunc, nLayer, nHiddenNeuron):

    # Build Neural Network
    nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                        activation = actFunc, verbose = False,
                        solver = 'lbfgs', learning_rate_init = 0.1,
                        max_iter = 10000, random_state = 20200408)
    fit = nn.fit(X, Y)

    # Test model
    pred = nn.predict_proba(X)

    # Call misclassification rate
    bad = 0
    for i in range(len(pred)):
        # print(pred[i])
        if pred[i][1] > pred[i][0] and Y[i] == 1:
            bad += 1
    err = bad / len(pred)

    iterations = nn.n_iter_
    loss = nn.loss_
    return err, iterations, loss, nn.out_activation_

# print('Layers\tNeurons\tActFn\tMCR\tLoss\tIter')

for act in ('identity', 'logistic', 'relu', 'tanh'):
    print('Using activation function %s...', act)
    row = None
    for layers in (1,2,3,4,5):
        for neurons in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            mcr, it, loss, ofn = Build_NN_Class (actFunc = act, nLayer = layers, nHiddenNeuron = neurons)
            r = (layers, neurons, mcr, loss, it, ofn)
            if ofn != 'logistic':
                print("!!!!!!!!!ofn", ofn)
            if row == None or (r[3] < row[3] and r[2] < row[2]):
                row = r
            # print("%s\t%s\t%s\t%s\t%f\t%s" % (layers, neurons, act[:4], mcr, loss, it))
    print("%s: %s" % (act, row))

# Build NN
nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (1,),
                    activation = 'relu', verbose = False,
                    solver = 'lbfgs', learning_rate_init = 0.1,
                    max_iter = 10000, random_state = 20200408)
fit = nn.fit(X, Y)

# Make scatterplot
colors = tuple(map(lambda g: ('red', 'blue')[g], map(lambda pred: 1 if pred[1] > pred[0] else 0, fit.predict_proba(X))))
# colors = tuple(map(lambda g: ('red', 'blue')[g], Y))   # ideal
plt.scatter(X['x'], X['y'], color=colors)
plt.xlabel('x')
plt.ylabel('y')
plt.title('relu NN classification of spiral')
plt.grid(True)
plt.show()