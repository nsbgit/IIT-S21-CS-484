# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:11:46 2021

@author: Sukanta Sharma
"""
import pandas as pd
import sklearn.svm as svm
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

entropyValue = 20210325


def customArcTan(z):
    theta = np.where(z < 0.0, 2.0 * np.pi + z, z)
    return theta


# ----------------------------------------------------------------------------
#                           Q2
# ----------------------------------------------------------------------------

data = pd.read_csv('SpiralWithCluster.csv', delimiter=",", usecols=["x", "y", "SpectralCluster"])
x_train = data[['x', 'y']]
y_train = data["SpectralCluster"]

# ----------------------------------------------------------------------------
#                           Q2.a
# ----------------------------------------------------------------------------

model = svm.SVC(kernel="linear", decision_function_shape="ovr", random_state=entropyValue)
thisFit = model.fit(x_train, y_train)
y_pred = thisFit.predict(x_train)
data['Predicted'] = y_pred
print("Intercept:", thisFit.intercept_)
print("Co-efficient:", thisFit.coef_)
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.b
# ----------------------------------------------------------------------------

misclassificationRate = 1 - metrics.accuracy_score(y_train, y_pred)
print('Misclassification Rate:', misclassificationRate)
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.c
# ----------------------------------------------------------------------------
intercept = thisFit.intercept_
x = np.linspace(-5, 5)
y = (-intercept - thisFit.coef_[0][0] * np.linspace(-5, 5)) / thisFit.coef_[0][1]
redSP = data[data['Predicted'] == 0]
plt.scatter(redSP['x'], redSP['y'], c="red", label=0)
blueSP = data[data['Predicted'] == 1]
plt.scatter(blueSP['x'], blueSP['y'], c="blue", label=1)
plt.plot(x, y, color='black', linestyle='dotted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM Scatter Plot Predicted Class with Hyperplane')
plt.grid(True)
plt.legend(title="Predicted Class")
plt.show()
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.d
# ----------------------------------------------------------------------------
data['radius'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
data['theta'] = np.arctan2(data['y'], data['x'])
data['theta'] = data['theta'].apply(customArcTan)
redSP = data[data['SpectralCluster'] == 0]
plt.scatter(redSP['radius'], redSP['theta'], c="red", label=0)
blueSP = data[data['SpectralCluster'] == 1]
plt.scatter(blueSP['radius'], blueSP['theta'], c="blue", label=1)
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM Scatter Plot for Polar Coordinates')
plt.grid(True)
plt.legend(title="Predicted Class")
plt.show()
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.e
# ----------------------------------------------------------------------------
data['group'] = 3
data.loc[data['radius'] <= 2.5, 'group'] = 2
data.loc[(data['radius'] < 3) & (data['theta'] >= 2), 'group'] = 2
data.loc[(data['radius'] < 3.5) & (data['theta'] >= 3), 'group'] = 2
data.loc[(data['radius'] < 4) & (data['theta'] >= 4), 'group'] = 2
data.loc[(data['radius'] < 2) & (data['theta'] >= 3), 'group'] = 1
data.loc[(data['radius'] < 2.5) & (data['theta'] >= 4), 'group'] = 1
data.loc[(data['radius'] < 3) & (data['theta'] >= 5), 'group'] = 1
data.loc[(data['radius'] < 1.5) & (data['theta'] >= 6), 'group'] = 0
redSP = data[data['group'] == 0]
plt.scatter(redSP['radius'], redSP['theta'], c="red", label=0)
blueSP = data[data['group'] == 1]
plt.scatter(blueSP['radius'], blueSP['theta'], c="blue", label=1)
green = data[data['group'] == 2]
plt.scatter(green['radius'], green['theta'], c="green", label=2)
black = data[data['group'] == 3]
plt.scatter(black['radius'], black['theta'], c="black", label=3)
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM Scatter Plot for Polar Coordinates having 4 Groups')
plt.grid(True)
plt.legend(title="Predicted Class", loc='lower right')
plt.show()
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.f
# ----------------------------------------------------------------------------
equations = []
for group in [(0, 1), (1, 2), (2, 3)]:
    groupedData = data[data['group'].isin(group)]
    x_train_group = groupedData[['radius', 'theta']]
    y_train_group = groupedData["group"]
    model = svm.SVC(kernel="linear", decision_function_shape="ovr", random_state=entropyValue)
    thisFit = model.fit(x_train_group, y_train_group)
    y_pred = thisFit.predict(x_train_group)

    print("SVM\n")
    print(group)
    print("Intercept:", thisFit.intercept_)
    print("Co-efficient:", thisFit.coef_)
    intercept = thisFit.intercept_
    x = np.linspace(0, 4.5)
    y = (-intercept - thisFit.coef_[0][0] * x) / thisFit.coef_[0][1]
    equations.append((x, y))
    print('*' * 10)
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.g
# ----------------------------------------------------------------------------
redSP = data[data['group'] == 0]
plt.scatter(redSP['radius'], redSP['theta'], c="red", label=0)
blueSP = data[data['group'] == 1]
plt.scatter(blueSP['radius'], blueSP['theta'], c="blue", label=1)
green = data[data['group'] == 2]
plt.scatter(green['radius'], green['theta'], c="green", label=2)
black = data[data['group'] == 3]
plt.scatter(black['radius'], black['theta'], c="black", label=3)
for e in equations:
    plt.plot(e[0], e[1], color='black', linestyle='dotted')
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM Scatter Plot for Polar Coordinates with Hyperplanes having 4 Groups')
plt.grid(True)
plt.legend(title="Predicted Class")
plt.show()
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.h
# ----------------------------------------------------------------------------
cartesianEquations = []
for e in equations:
    cartesianEquations.append((e[0] * np.cos(e[1]), e[0] * np.sin(e[1])))
redSP = data[data['SpectralCluster'] == 0]
plt.scatter(redSP['x'], redSP['y'], c="red", label=0)
blueSP = data[data['SpectralCluster'] == 1]
plt.scatter(blueSP['x'], blueSP['y'], c="blue", label=1)
count = 0
for e in cartesianEquations:
    if count == 0:
        clr = 'green'
    else:
        clr = 'black'
    plt.plot(e[0], e[1], color=clr, linestyle='dotted')
    count += 1
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM Scatter Plot for Cartesian Coordinates with Hyperplanes')
plt.grid(True)
plt.legend(title="Predicted Class")
plt.show()
print('-' * 40, end='\n')