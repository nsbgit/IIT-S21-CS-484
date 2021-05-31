# Load the PANDAS library
import pandas
fraud = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\Fraud.csv',
                        delimiter=',')

# Examine a portion of the data frame
print(fraud)

# Put the descriptive statistics into another dataframe
fraud_description = fraud.groupby('FRAUD').describe()

# Visualize the boxplot of the DEBTINC variable by BAD
import matplotlib.pyplot as plt

fraud.boxplot(column='TOTAL_SPEND', by='FRAUD', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("TOTAL_SPEND")
plt.ylabel("FRAUD")
plt.show()

fraud.boxplot(column='DOCTOR_VISITS', by='FRAUD', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("DOCTOR_VISITS")
plt.ylabel("FRAUD")
plt.show()

fraud.boxplot(column='NUM_CLAIMS', by='FRAUD', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("NUM_CLAIMS")
plt.ylabel("FRAUD")
plt.show()

fraud.boxplot(column='MEMBER_DURATION', by='FRAUD', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("MEMBER_DURATION")
plt.ylabel("FRAUD")
plt.show()

fraud.boxplot(column='OPTOM_PRESC', by='FRAUD', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("OPTOM_PRESC")
plt.ylabel("FRAUD")
plt.show()

fraud.boxplot(column='NUM_MEMBERS', by='FRAUD', vert=False, figsize=(6,4))
plt.suptitle("")
plt.title("")
plt.xlabel("NUM_MEMBERS")
plt.ylabel("FRAUD")
plt.show()

# Specify the target label
target = fraud[['FRAUD']]

# Orthonormalize the input fields
import numpy as np
from numpy import linalg as LA

inputField = fraud[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']]

# Convert the DataFrame into numpy array for easier manipulation
x = inputField.values
xtx = np.matmul(x.transpose(), x)
evals, evecs = LA.eigh(xtx)
transf = np.matmul(evecs, LA.inv(np.sqrt(np.diagflat(evals))))
transf_x = np.matmul(x, transf)

# Show that the training data is actually orthnormalize
check = np.matmul(transf_x.transpose(), transf_x)
print("Does this look like an identity matrix? \n", check)

# Find the nearest neighbors
trainData = pandas.DataFrame(transf_x)

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

kNNSpec = KNeighborsClassifier(n_neighbors=5, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData, np.ravel(target))
distances, indices = nbrs.kneighbors(trainData)
prediction = nbrs.predict(trainData)
score_result = nbrs.score(trainData, target)

# Find the neighbors of a specific row
focal = [[7500., 15., 3., 127., 2., 2.]]
transf_xf = np.matmul(focal, transf)
dist_f, index_f = nbrs.kneighbors(pandas.DataFrame(transf_xf))

for i in index_f:
    print("Neighbor Value: \n", x[i])
    print("Index and FRAUD: \n", target.iloc[i])
