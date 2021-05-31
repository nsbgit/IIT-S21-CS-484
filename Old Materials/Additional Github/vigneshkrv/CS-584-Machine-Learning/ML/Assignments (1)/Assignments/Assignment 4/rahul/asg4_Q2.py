import numpy as np
import pandas as pd

import sklearn.naive_bayes as NB

dataframe =  pd.read_csv("Purchase_Likelihood.csv")

X = pd.DataFrame(dataframe, columns=['group_size','homeowner', 'married_couple'])
y = pd.DataFrame(dataframe, columns=['A'])

classifier = NB.MultinomialNB().fit(X, y)

print('Log Class Probability:\n', classifier.class_log_prior_ )
print('Class Probability:\n', np.exp(classifier.class_log_prior_ ))

predProb = classifier.predict_proba(X)
predProbX = pd.DataFrame(predProb, columns=["P(A=0)", "P(A=1)", "P(A=2)"])


X_test = np.array([[1,0,0], [2,1,1], [3,1,1], [4,0,0]])
X_test = pd.DataFrame(X_test, columns=['group_size','homeowner', 'married_couple'])
predProbTest = pd.DataFrame(classifier.predict_proba(X_test), columns=["P(A=0)", "P(A=1)", "P(A=2)"])

finalTestResults = pd.concat([X_test, predProbTest], axis=1)

print(finalTestResults)

finalResults = pd.concat([X, predProbX], axis=1)

print(finalResults)

print(finalResults.loc[finalResults['P(A=1)'].idxmax()])

