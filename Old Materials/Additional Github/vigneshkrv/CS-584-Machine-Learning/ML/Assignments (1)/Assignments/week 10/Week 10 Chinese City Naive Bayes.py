import numpy
import pandas

import sklearn.naive_bayes as NB

X = numpy.array([[2,1,0,0,0,0],
                 [2,0,1,0,0,0],
                 [1,0,0,1,0,0],
                 [1,0,0,0,1,1]])

y = numpy.array([0,0,0,1])

classifier = NB.MultinomialNB().fit(X, y)

print('Class Count:\n', classifier.class_count_)
print('Log Class Probability:\n', classifier.class_log_prior_ )
print('Feature Count (after adding alpha):\n', classifier.feature_count_)
print('Log Feature Probability:\n', classifier.feature_log_prob_)

predProb = classifier.predict_proba(X)
print('Predicted Conditional Probability (Training):', predProb)

X_test = numpy.array([[3,0,0,0,1,1],
                      [0,1,1,0,1,1]])

print('Predicted Conditional Probability (Testing):\n', classifier.predict_proba(X_test))
