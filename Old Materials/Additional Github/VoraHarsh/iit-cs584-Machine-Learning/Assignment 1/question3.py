import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from numpy import linalg as la

fraud_df = pd.read_csv('Fraud.csv', index_col='CASE_ID')

N = len(fraud_df.index)

fraud_list = fraud_df[fraud_df.FRAUD == 1]
not_fraud_list = fraud_df[fraud_df.FRAUD == 0]

# Question 3.a

fraud_percent = (len(fraud_list)/N) * 100
print("What percent of investigations are found to be fraudulent? Please give your answer up to 4 decimal places.")
print(np.round(fraud_percent,4), "% of investigations are found to be fraudulent.")
print()

# Question 3.b
# To check graph for each interval variable please uncomment the next one and comment the previous one.

fig = plt.figure()
ax = fig.add_subplot(111)

ax.boxplot([fraud_list["TOTAL_SPEND"],not_fraud_list["TOTAL_SPEND"]], vert=False)
plt.title("Box Plot of Total Spend")
plt.grid(True)
ax.set_yticklabels(['Fraudulent', 'Otherwise'])
plt.xlabel("X values")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([fraud_list["DOCTOR_VISITS"],not_fraud_list["DOCTOR_VISITS"]], vert=False)
plt.title("Box Plot of Doctor Visits")
plt.grid(True)
ax.set_yticklabels(['Fraudulent', 'Otherwise'])
plt.xlabel("X values")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([fraud_list["NUM_CLAIMS"],not_fraud_list["NUM_CLAIMS"]], vert=False)
plt.title("Box Plot of Number of Claims")
plt.grid(True)
ax.set_yticklabels(['Fraudulent', 'Otherwise'])
plt.xlabel("X values")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([fraud_list["MEMBER_DURATION"],not_fraud_list["MEMBER_DURATION"]], vert=False)
plt.title("Box Plot of Membership Duration in months")
plt.grid(True)
ax.set_yticklabels(['Fraudulent', 'Otherwise'])
plt.xlabel("X values")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([fraud_list["OPTOM_PRESC"],not_fraud_list["OPTOM_PRESC"]], vert=False)
plt.title("Box Plot of Number of Optical Examinations")
plt.grid(True)
ax.set_yticklabels(['Fraudulent', 'Otherwise'])
plt.xlabel("X values")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([fraud_list["NUM_MEMBERS"],not_fraud_list["NUM_MEMBERS"]], vert=False)
plt.title("Box Plot of Number of Members covered")
plt.grid(True)
ax.set_yticklabels(['Fraudulent', 'Otherwise'])
plt.xlabel("X values")
plt.show()


# Question 3.c

without_fraud = fraud_df.drop(columns=['FRAUD'])

matrix_X = np.matrix(without_fraud.values)

xtx = matrix_X.transpose() * matrix_X

evals, evecs = np.linalg.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)
print()
print("Proof that 6 dimensions are used and eigen values of all is greater then 1")
print(evals>1)
print()

print("Please provide the transformation matrix?  You must provide proof that the resulting variables are actually orthonormal.")
transf = evecs * la.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)
print()

transf_x = matrix_X * transf;
#print("The Transformed x = \n", transf_x)

xtx = transf_x.transpose() * transf_x
print("Proof that the resulting variables are actually orthonormal.")
print("Expect an Identity Matrix = \n", xtx)
print()

#Question 3.d

target = fraud_df.iloc[:,0]
neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(transf_x, target)

print("Score Function result:{}".format(nbrs.score(transf_x, target)))
print()

# Question 3.e

focal = [[7500, 15, 3, 127, 2, 2]]
trans_focal = focal * transf
print("Transformed Input Variables: ",trans_focal)
myNeighbors = nbrs.kneighbors(trans_focal, return_distance = False)
print("Neighbors: ", myNeighbors)
print()

# Question 3.f

class_prob = nbrs.predict_proba(trans_focal)
class_predict = nbrs.predict(trans_focal)

print("Predicted label for Test Data:",(class_predict))
print("Probability values of test set:",(class_prob))

