# load necessary libraries
import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as metrics
import pandas as pd
from sklearn.model_selection import train_test_split


def predict_class(in_data):
    if in_data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if in_data['EDUCATION'] <= 0.5:
            return [0.6832518880497557, 0.3167481119502443]
        else:
            return [0.8912579957356077, 0.10874200426439233]
    else:
        if in_data['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.006838669567920423, 0.9931613304320795]
        else:
            return [0.5306122448979592, 0.46938775510204084]


def predict_class_decision_tree(in_data):
    out_data = numpy.ndarray(shape=(len(in_data), 2), dtype=float)
    counter = 0
    for index, row in in_data.iterrows():
        probability = predict_class(in_data=row)
        out_data[counter] = probability
        counter += 1
    return out_data


print("=" * 50)
print("=" * 50)
print("ML-Assignment 3-Question 3")
print("=" * 50)
print("=" * 50)
# Please apply your decision tree to the Test partition and then provide the following information.

# loading data from data file
claim_history_data = pd.read_csv('claim_history.csv', delimiter=',')
claim_history_data = claim_history_data[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]].dropna()

# splitting data into training and testing part
p_training, p_testing = 0.7, 0.3
claim_history_data_train, claim_history_data_test = train_test_split(claim_history_data, train_size=p_training,
                                                                     test_size=p_testing, random_state=27513)
train_data = claim_history_data_train
test_data = claim_history_data_test
# separate input and output variables in both part
train_data_x = train_data[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
train_data_y = train_data['CAR_USE']
test_data_x = test_data[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
test_data_y = test_data["CAR_USE"]

test_data_x['EDUCATION'] = test_data_x['EDUCATION'].map(
    {'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

print()
print("=" * 50)
print("ML-Assignment 3-Question 3-Section a)")
print("=" * 50)
# a)	(10 points). Use the proportion of target Event value in the training partition as the threshold, what is the
# Misclassification Rate in the Test partition?

# calculating threshold
threshold = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]

# predict probability for testing input data
pred_prob_y = predict_class_decision_tree(in_data=test_data_x)
pred_prob_y = pred_prob_y[:, 0]

target_y = test_data_y
num_y = target_y.shape[0]

# determining the predicted class
pred_y = numpy.empty_like(target_y)
for i in range(num_y):
    if pred_prob_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'

# calculating accuracy and misclassification rate
accuracy = metrics.accuracy_score(target_y, pred_y)
misclassification_rate = 1 - accuracy
print(f'Accuracy: {accuracy}')
print(f'Misclassification Rate: {misclassification_rate}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 3-Section b)")
print("=" * 50)
# b)     (10 points). What is the Root Average Squared Error in the Test partition?

# calculating the root average squared error
RASE = 0.0
for y, ppy in zip(target_y, pred_prob_y):
    if y == 'Commercial':
        RASE += (1 - ppy) ** 2
    else:
        RASE += (0 - ppy) ** 2
RASE = numpy.sqrt(RASE / num_y)
print(f'Root Average Squared Error: {RASE}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 3-Section c)")
print("=" * 50)
# c)    (10 points). What is the Area Under Curve in the Test partition?
y_true = 1.0 * numpy.isin(target_y, ['Commercial'])
AUC = metrics.roc_auc_score(y_true, pred_prob_y)
print(f'Area Under Curve: {AUC}')

print()
print("=" * 50)
print("ML-Assignment 3-Question 3-Section d)")
print("=" * 50)
# d)	(10 points). Generate the Receiver Operating Characteristic curve for the Test partition.  The axes must be
# properly labeled.  Also, don’t forget the diagonal reference line.

# generating the coordinates for the roc curve
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(target_y, pred_prob_y, pos_label='Commercial')

# adding two dummy coordinates
one_minus_specificity = numpy.append([0], one_minus_specificity)
sensitivity = numpy.append([0], sensitivity)

one_minus_specificity = numpy.append(one_minus_specificity, [1])
sensitivity = numpy.append(sensitivity, [1])

# drawing the roc curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
