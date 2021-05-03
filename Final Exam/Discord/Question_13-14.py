import pandas as pd
from sklearn import naive_bayes
pd.options.mode.chained_assignment = None  # default='warn'


credit = pd.read_csv('Q11.csv')

x = credit[['Gender', 'MaritalStatus', 'Retired']]
idx = x.index[(x['Gender']=='Female') & (x['MaritalStatus']=='Unmarried') & 
              (x['Retired'] == 'Yes')]
y = credit[['CreditCard']]
x.Gender[x.Gender == 'Male'] = 1
x.Gender[x.Gender == 'Female'] = 0
x.MaritalStatus[x.MaritalStatus == 'Unmarried'] = 0
x.MaritalStatus[x.MaritalStatus == 'Married'] = 1
x.Retired[x.Retired == 'Yes'] = 1
x.Retired[x.Retired == 'No'] = 0
# 0 == American Express
# 1 == Discover
# 2 == Mastercard
# 3 == Visa
# 4 == Others
y.CreditCard[y.CreditCard == 'American Express'] = 0
y.CreditCard[y.CreditCard == 'Discover'] = 1
y.CreditCard[y.CreditCard == 'MasterCard'] = 2
y.CreditCard[y.CreditCard == 'Visa'] = 3
y.CreditCard[y.CreditCard == 'Others'] = 4
y = y.squeeze().astype('category')
x = x.astype('category')

# Gender = Female, MaritalStatus = Unmarried, and Retired = Yes
idx1 = x[(x['Gender']==0) & (x['MaritalStatus']==0) & 
              (x['Retired'] == 1)]
_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(x, y)
pred_prob1 = _objNB.predict_proba(idx1)
print('pred_prob1[0][0] = ', pred_prob1[0][0])

#Gender = Female, MaritalStatus = Unmarried, and Retired = No
idx2 = x[(x['Gender']==0) & (x['MaritalStatus']==0) & 
              (x['Retired'] == 0)]
_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(x, y)
pred_prob2 = _objNB.predict_proba(idx2)
print('pred_prob2[0][0] = ', pred_prob2[0][0])

# Gender = Male, MaritalStatus = Married, and Retired = Yes 
idx3 = x[(x['Gender']==1) & (x['MaritalStatus']==1) & 
              (x['Retired'] == 1)]
_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(x, y)
pred_prob3 = _objNB.predict_proba(idx3)
print('pred_prob3[0][0] = ', pred_prob3[0][0])

# Gender = Male, MaritalStatus = Married, and Retired = No
idx4 = x[(x['Gender']==1) & (x['MaritalStatus']==1) & 
              (x['Retired'] == 0)]
_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(x, y)
pred_prob4 = _objNB.predict_proba(idx4)
print('pred_prob4[0][0] = ', pred_prob4[0][0])

# Gender = Female, MaritalStatus = Married, and Retired = No
idx5 = x[(x['Gender']==0) & (x['MaritalStatus']==1) & 
              (x['Retired'] == 0)]
_objNB = naive_bayes.CategoricalNB(alpha = 1.0e-10)
thisModel = _objNB.fit(x, y)
pred_prob5 = _objNB.predict_proba(idx5)
print('pred_prob5[0][0] = ', pred_prob5[0][0])

print('highest probability is for d') 
