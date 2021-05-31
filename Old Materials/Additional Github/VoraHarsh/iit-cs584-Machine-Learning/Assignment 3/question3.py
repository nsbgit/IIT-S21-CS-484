import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.model_selection import train_test_split

def predict_class(data):
    if data['OCCUPATION'] in ('Blue Collar', 'Student', 'Unknown'):
        if data['EDUCATION'] <= 0.5:
            return [0.2693548387096774, 0.7306451612903225]
        else:
            return [0.8376594808622966, 0.16234051913770348]
    else:
        if data['CAR_TYPE'] in ('Minivan', 'SUV', 'Sports Car'):
            return [0.008420441347270616, 0.9915795586527294]
        else:
            return [0.5341972642188625, 0.4658027357811375]

def predict_class_decision_tree(data):
    outputdata = np.ndarray(shape=(len(data), 2), dtype=float)
    counter = 0
    for index, row in data.iterrows():
        probability = predict_class(data=row)
        outputdata[counter] = probability
        counter += 1
    return outputdata

dataframe = pd.read_csv('claim_history.csv', delimiter=',')
dataframe = dataframe[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]]

dataframe['EDUCATION_VAL'] = dataframe['EDUCATION'].map({'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

training_data = 0.75
testing_data = 0.25
train_data, test_data = train_test_split(dataframe, train_size=training_data, test_size=testing_data, random_state=60616, stratify=dataframe["CAR_USE"])

train_x = train_data[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
train_y = train_data['CAR_USE']
test_x = test_data[["CAR_TYPE", "OCCUPATION", "EDUCATION"]]
test_y = test_data["CAR_USE"]

test_x['EDUCATION'] = test_x['EDUCATION'].map({'Below High School': 0, 'High School': 1, 'Bachelors': 2, 'Masters': 3, 'Doctors': 4})

#Question 1

print("(10 points). Use the proportion of target Event value in the training partition as the threshold, what is the Misclassification Rate in the Test partition?")

# calculating threshold
threshold = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]

# predict probability for testing input data
pred_prob_y = predict_class_decision_tree(data=test_x)
pred_prob_y = pred_prob_y[:, 0]

# determining the predicted class
pred_y = np.empty_like(test_y)
for i in range(test_y.shape[0]):
    if pred_prob_y[i] > threshold:
        pred_y[i] = 'Commercial'
    else:
        pred_y[i] = 'Private'

# Calculating accuracy and Misclassification rate
accuracy = metrics.accuracy_score(test_y, pred_y)
misclassification_rate = 1 - accuracy
print("Accuracy: ",accuracy)
print("Misclassification Rate:",misclassification_rate)
print()


#Question 2
print("(5 points). Use the Kolmogorov-Smirnov event probability cutoff value in the training partition as the threshold, what is the Misclassification Rate in the Test partition?")

ks_threshold = 0.5341972642188625
predicted_y = np.empty_like(test_y)
for i in range(test_y.shape[0]):
    if pred_prob_y[i] > ks_threshold:
        predicted_y[i] = 'Commercial'
    else:
        predicted_y[i] = 'Private'

# Calculating accuracy and Misclassification rate
ks_accuracy= metrics.accuracy_score(test_y, predicted_y)
ks_misclassification_rate = 1 - ks_accuracy
print("Accuracy: ",ks_accuracy)
print("Misclassification Rate:",ks_misclassification_rate)
print()

#Question 3

print("(5 points). What is the Root Average Squared Error in the Test partition?")
# Calculate the Root Average Squared Error
RASE = 0.0
for y, ppy in zip(test_y, pred_prob_y):
    if y == 'Commercial':
        RASE += (1 - ppy) ** 2
    else:
        RASE += (0 - ppy) ** 2
RASE = np.sqrt(RASE / test_y.shape[0])
print("Root Average Squared Error in the Test partition is ",RASE)
print()

#Question 4
print("(5 points). What is the Area Under Curve in the Test partition?")
y_true = 1.0 * np.isin(test_y, ['Commercial'])
#AUC = metrics.roc_auc_score(y_true, pred_prob_y)
#print('Area Under Curve: ', AUC)

Events = []
NonEvents = []

for i in range(len(y_true)):
    if y_true[i]==1:
        Events.append(pred_prob_y[i])
    else:
        NonEvents.append(pred_prob_y[i])

ConcordantPairs = 0
DiscordantPairs = 0
TiedPairs = 0

for i in Events:
    for j in NonEvents:
        if i>j:
            ConcordantPairs = ConcordantPairs + 1
        elif i<j:
            DiscordantPairs = DiscordantPairs + 1
        else:
            TiedPairs = TiedPairs + 1

AUC = 0.5 + 0.5 * ((ConcordantPairs-DiscordantPairs)/(ConcordantPairs+DiscordantPairs+TiedPairs))
print("Area Under Curve in the Test Partition is ", AUC)
print()

#Question 5

print("(5 points). What is the Gini Coefficient in the Test partition?")
#GINI = 2 * AUC - 1
#print("Gini Coefficient is",GINI)
#print()

gini = ((ConcordantPairs-DiscordantPairs)/(ConcordantPairs+DiscordantPairs+TiedPairs))
print("Gini coefficient in Test Partition is ",gini)
print()

#Question 6

print("(5 points). What is the Goodman-Kruskal Gamma statistic in the Test partition?")

goodman_coefficient = ((ConcordantPairs-DiscordantPairs)/(ConcordantPairs+DiscordantPairs))
print("Goodman-Kruskal Gamma statistic in the Test partition is ",goodman_coefficient)
print()

#Question 7
print("(10 points). Generate the Receiver Operating Characteristic curve for the Test partition.  The axes must be properly labeled.  Also, don’t forget the diagonal reference line.")
# generating the coordinates for the roc curve
one_minus_specificity, sensitivity, thresholds = metrics.roc_curve(test_y, pred_prob_y, pos_label='Commercial')

# adding two dummy coordinates
one_minus_specificity = np.append([0], one_minus_specificity)
sensitivity = np.append([0], sensitivity)

one_minus_specificity = np.append(one_minus_specificity, [1])
sensitivity = np.append(sensitivity, [1])

# plotting the roc curve
plt.figure(figsize=(6, 6))
plt.plot(one_minus_specificity, sensitivity, marker='o', color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()