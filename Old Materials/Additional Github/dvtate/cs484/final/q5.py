
import numpy
import pandas
import sklearn
import sklearn.linear_model
import sklearn.metrics
from sklearn.model_selection import train_test_split

# Get data
df = pandas.read_csv('FinalQ5.csv')
X = df[['TransportMode', 'CommuteMile']]
Y = df['Late4Work']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=42)

model = sklearn.linear_model.LogisticRegression(
    max_iter=10000,
    verbose=True,
    random_state=0).fit(X_train, y_train)

print('score: ', model.score(X_test, y_test))
print('inter: ', model.intercept_)
print('iter: ', model.n_iter_)
print('iter: ', model.coef_)
