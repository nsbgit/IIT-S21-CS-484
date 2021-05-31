import pandas
from sklearn.model_selection import train_test_split

hmeq = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',')

print('Number of Observations = ', hmeq.shape[0])
print(hmeq.groupby('BAD').size() / hmeq.shape[0])

hmeq_train, hmeq_test = train_test_split(hmeq, test_size = 0.3, random_state = 60616)

print('Number of Observations in Training = ', hmeq_train.shape[0])
print('Number of Observations in Testing = ', hmeq_test.shape[0])

print(hmeq_train.groupby('BAD').size() / hmeq_train.shape[0])

print(hmeq_test.groupby('BAD').size() / hmeq_test.shape[0])


hmeq_train, hmeq_test = train_test_split(hmeq, test_size = 0.3, random_state = 60616, stratify = hmeq['BAD'])

print('Number of Observations in Training = ', hmeq_train.shape[0])
print('Number of Observations in Testing = ', hmeq_test.shape[0])

print(hmeq_train.groupby('BAD').size() / hmeq_train.shape[0])

print(hmeq_test.groupby('BAD').size() / hmeq_test.shape[0])