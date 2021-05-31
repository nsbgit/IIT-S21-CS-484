# imports
import pandas as pd
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import math
import functools

# Load data
df = pd.read_csv('Fraud.csv')

############
# Q3.a
############
print('Q3.a')

# Calculate fraud rate
fraud_n = functools.reduce(lambda a, v: a + v, df['FRAUD'], 0)
fraud_rate = fraud_n / len(df['FRAUD'])
print('\tfraudulent accounts:', fraud_n)
print('\ttotal accounts:', len(df['FRAUD']))
print("\tfraud rate: %3.4f%%" % (100 * fraud_rate))

############
# Q3.b
############
print('\nQ3.b\n\tsee charts')

# Box plot for each field
for field in df:
    # Skip external fields
    if field in ('CASE_ID', 'FRAUD'):
        continue

    # Draw boxplot
    df.boxplot(column=field, by='FRAUD', vert=False, whis=1.5)
    plt.title("Boxplot of %s by levels of FRAUD" % field)
    plt.xlabel(field)
    plt.ylabel('fraud')
    plt.grid(axis="y")
    plt.show()

############
# Q3.c
############
print('\nQ3.c')

# Orthonormalization

# get interval varaible fields
fields = [field for field in df if field not in ('CASE_ID', 'FRAUD')]

# transpose
# mattx = mat.transpose() * mat
data = df.set_index('CASE_ID')
mat = numpy.matrix(df.values)
mattx = mat.transpose() * mat
print('t(data) * data:\n', mattx)

# Eigenvalue decomposition
evals, evecs = numpy.linalg.eigh(mattx)
usable_field_pairs = list(filter(lambda p: p[1] > 1, zip(fields, evals)))
usable_fields = list(map(lambda p: p[0], usable_field_pairs))
print("eigenvalues:\n", evals)
print('usable dimensions:')
for f, ev in usable_field_pairs:
    print('\t', f, ':', ev)

# Here is the transformation matrix
transf = evecs * numpy.linalg.inv(numpy.sqrt(numpy.diagflat(evals)))
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_df = mat * transf
print("The Transformed x = \n", transf_df)

# Check columns of transformed X
print("Expect an Identity Matrix = \n", numpy.round(transf_df.transpose() * transf_df, 3))

############
# Q3.d
############
print('\nQ3.d')

from sklearn.neighbors import KNeighborsClassifier

# Perform classification
trainData = data[usable_fields]
target = data['FRAUD']
neigh = KNeighborsClassifier(n_neighbors=4 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)

# See the classification probabilities
class_prob = nbrs.predict_proba(trainData)
print('classification probabilities:\n', class_prob)

# Test performance with our training dataset
test_data = data[[field for field in data if field not in ('FRAUD')]]
print('score:', neigh.score(test_data, target))
print('This is the ratio of correct predictions over total number of tests for our training data')

###############
# Q3.e
###############
print('\nQ3.e')

# Find nearest neighbor
investigation = [[
    7500, 
    15, 
    3, 
    127, 
    2, 
    2,
]]
print('kneighbors:\n', nbrs.kneighbors(investigation, n_neighbors=5))

##############
# Q3.f
##############
print('\nQ3.f')

# Make prediction
print('prediction:', nbrs.predict(investigation))
print('prediction probability:', nbrs.predict_proba(investigation))
