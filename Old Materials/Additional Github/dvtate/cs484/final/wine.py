import pandas
import numpy
import sklearn
import sklearn.neighbors
import sklearn.linear_model
import sklearn.metrics
import sklearn.svm
import sklearn.neural_network

# Load data
train = pandas.read_csv('WineQuality_Train.csv')
test = pandas.read_csv('WineQuality_Test.csv')
Y_train = train['quality_grp']
Y_test = test['quality_grp']
X_train = train[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]
X_test = test[['alcohol','citric_acid','free_sulfur_dioxide','residual_sugar','sulphates']]


# Logistic regression
lr_m = sklearn.linear_model.LogisticRegression(
    solver='newton-cg',
    max_iter=10000000000000,
    multi_class='multinomial',
    random_state=0
    ).fit(X_train, Y_train)
print('logistic regression score: ', lr_m.score(X_test, Y_test))


# SVM Model
# svm_m = sklearn.svm.SVC(
#     kernel = 'linear',
#     random_state = 20201202,
#     max_iter = -1,
#     probability=False,
#     # verbose=True,
#     ).fit(X_train, Y_train)
# print('svm score: ', svm_m.score(X_test, Y_test))



# Util to make nn classifier
def do_nn(nLayer, nHiddenNeuron):

    # Build Neural Network
    nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                        activation = 'relu', verbose = False,
                        solver = 'lbfgs', learning_rate_init = 0.1,
                        max_iter = 5000, random_state = 20201202)
    fit = nn.fit(X_train, Y_train)
    return nn.loss_, 1 - fit.score(X_test, Y_test)


# print('Layers\tNeurons\tActFn\tMCR\tLoss\tIter')
print('l n loss mcr')
ts = []
for layers in (1,2,3,4,5,6,7,8,9,10):
    for neurons in (5, 6, 7, 8, 9, 10):
        loss, mcr = do_nn(nLayer = layers, nHiddenNeuron = neurons)
        row = (layers, neurons, loss, mcr)
        ts.append(row)
        # print('%s\t%s\t%s\t%s' % row)

print('layers', 'neurons', 'loss', 'missclassification rate');
ts.sort(key=lambda t: t[2])
for r in ts[:10]:
    print(r)
