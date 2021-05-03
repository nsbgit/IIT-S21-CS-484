# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:16:46 2021

@author: Sukanta
"""

# Importig Libraries
import numpy
import pandas
import sklearn.tree as tree
import sklearn.metrics as metrics

# -------------------------------

# global variables
SPLITTING_CRITERION = 'entropy'
MAXIMUM_TREE_DEPTH = 5
INIT_RNDM_SEED = 20210415
MAX_BOOSTING_ITR = 50
INTERRUPT_ACCURACY = 0.9999999
INPUT_FEATURES = ['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']
TARGET_FEATURE = 'quality_grp'
# -------------------------------

# Read Data -------------------------------
train_Data = pandas.read_csv('WineQuality_Train.csv')
test_Data = pandas.read_csv('WineQuality_Test.csv')
# -------------------------------
nObs = train_Data.shape[0]

x_train = train_Data[INPUT_FEATURES]
y_train = train_Data[TARGET_FEATURE]

x_test = train_Data[INPUT_FEATURES]
y_test = train_Data[TARGET_FEATURE]

# q1.a, b -------------------------------


w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(MAX_BOOSTING_ITR)
ensemble_predicted_probability = numpy.zeros((nObs, 2))
is_converged = False
coverged_accuracy = numpy.nan
covereged_iteration = numpy.nan

for i in range(MAX_BOOSTING_ITR):
    classification_tree = tree.DecisionTreeClassifier(criterion=SPLITTING_CRITERION, max_depth=MAXIMUM_TREE_DEPTH,
                                                      random_state=INIT_RNDM_SEED)
    tree_fit = classification_tree.fit(x_train, y_train, w_train)
    tree_predicted_probability = classification_tree.predict_proba(x_train)
    accuracy[i] = classification_tree.score(x_train, y_train, w_train)
    ensemble_predicted_probability += accuracy[i] * tree_predicted_probability

    if accuracy[i] >= INTERRUPT_ACCURACY:
        is_converged = True
        coverged_accuracy = accuracy[i]
        covereged_iteration = i
        break

    # Update the weights
    event_error = numpy.where(
        y_train == 1
        , (1 - tree_predicted_probability[:, 1])
        , (tree_predicted_probability[:, 1]))
    predicted_class = numpy.where(tree_predicted_probability[:, 1] >= 0.2, 1, 0)
    w_train = numpy.where(predicted_class != y_train, 2 + numpy.abs(event_error), numpy.abs(event_error))

misclassification_rate = 1 - accuracy

print(f'The Misclassification Rate of the classification tree on the Training data at Iteration  0 = {misclassification_rate[0]}')
print(f'The Misclassification Rate of the classification tree on the Training data at Iteration  1 = {misclassification_rate[1]}')

#  q1.c -------------------------------
if is_converged:
    print(
        'The iteration is converged at Iteration {} and the Misclassification Rate of the classification tree on the Training data is {:.7f}'.format(
            covereged_iteration, 1 - coverged_accuracy))
else:
    print('The iteration is not converged')

# q1.d -------------------------------


y_score = tree_fit.predict_proba(x_test)
auc = metrics.roc_auc_score(y_test, y_score[:, 1])
print(f'The Area Under Curve metric on the Testing data using the final converged classification tree is {auc}')

# q1.e -------------------------------


df_box = pandas.concat(
    [pandas.DataFrame(y_test), pandas.DataFrame(tree_predicted_probability[:, 1])]
    , axis=1
    , join='inner')
df_box = df_box.rename(columns={0: "Prediction"})
df_box.boxplot(column='Prediction', by='quality_grp')
