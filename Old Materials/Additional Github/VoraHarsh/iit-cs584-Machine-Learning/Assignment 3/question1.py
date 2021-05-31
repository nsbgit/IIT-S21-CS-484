import pandas as pd
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('claim_history.csv', delimiter=',')
dataframe = dataframe[["CAR_TYPE", "OCCUPATION", "EDUCATION", "CAR_USE"]]

training_data = 0.75
testing_data = 0.25
train_data, test_data = train_test_split(dataframe, train_size=training_data, test_size=testing_data, random_state=60616, stratify=dataframe["CAR_USE"])

print('Total number of training data = ', train_data.shape[0])
print('Total number of testing data = ', test_data.shape[0])
print()

#Question 1.a)
print("(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Training partition?")
train_table = train_data["CAR_USE"].value_counts().to_frame().reset_index()
train_table.columns = ['CAR_USE','COUNT']
train_table['PROPORTION'] = train_table['COUNT'] / train_data["CAR_USE"].shape[0]
print("Frequency Table of the target variable in Training Partition: \n",train_table)
print()

#Question 1.b)
print("(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Test partition?")
train_table = test_data["CAR_USE"].value_counts().to_frame().reset_index()
train_table.columns = ['CAR_USE','COUNT']
train_table['PROPORTION'] = train_table['COUNT'] / test_data["CAR_USE"].shape[0]
print("Frequency Table of the target variable in Test Partition: \n",train_table)
print()

#Question 1.c)
print("(5 points). What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?")
training_prob = train_data.groupby("CAR_USE").size()["Commercial"] / train_data.shape[0]
testing_prob = test_data.groupby("CAR_USE").size()["Commercial"] / test_data.shape[0]
prob_comm = (training_prob * training_data) + (testing_prob * testing_data)
prob_training_comm = (training_prob * training_data) / prob_comm
print("Probability that an observation is in the Training partition given that CAR_USE = Commercial is:", prob_training_comm)
print()

#Question 1.d)
print("(5 points). What is the probability that an observation is in the Test partition given that CAR_USE = Private?")
testing_prob = test_data.groupby("CAR_USE").size()["Private"] / test_data.shape[0]
training_prob = train_data.groupby("CAR_USE").size()["Private"] / train_data.shape[0]
prob_private = (testing_prob * testing_data) + (training_prob * training_data)
prob_testing_private = (testing_prob * testing_data) / prob_private
print("Probability that an observation is in the Test partition given that CAR_USE = Private is: ",prob_testing_private)
