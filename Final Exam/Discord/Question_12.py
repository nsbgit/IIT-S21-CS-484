import pandas as pd
from sklearn.linear_model import LogisticRegression
pd.options.mode.chained_assignment = None  # default='warn'


credit = pd.read_csv('Q11.csv')

x = credit[['Gender', 'MaritalStatus', 'Retired']]
idx = x.index[(x['Gender']=='Female') & (x['MaritalStatus']=='Unmarried') & 
              (x['Retired'] == 'Yes')]
print('idx = ', idx)
y = credit[['CreditCard']]
x.Gender[x.Gender == 'Male'] = 1
x.Gender[x.Gender == 'Female'] = 0
x.MaritalStatus[x.MaritalStatus == 'Unmarried'] = 0
x.MaritalStatus[x.MaritalStatus == 'Married'] = 1
x.Retired[x.Retired == 'Yes'] = 1
x.Retired[x.Retired == 'No'] = 0
y.CreditCard[y.CreditCard == 'American Express'] = 0
y.CreditCard[y.CreditCard == 'Discover'] = 1
y.CreditCard[y.CreditCard == 'MasterCard'] = 2
y.CreditCard[y.CreditCard == 'Visa'] = 3
y.CreditCard[y.CreditCard == 'Others'] = 4
y = y.squeeze().astype('category')
x = x.astype('category')

idx = x[(x['Gender']==0) & (x['MaritalStatus']==0) & 
              (x['Retired'] == 1)]

logistic = LogisticRegression(random_state = 20210415).fit(x,y)
pred_prob = logistic.predict_proba(idx)

# 0 == American Express
# 1 == Discover
# 2 == Mastercard
# 3 == Visa
# 4 == Others
