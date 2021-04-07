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
# import sklearn.neural_network as nn


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


# ----------------------------------------------------------------------------
#                           Q2
# ----------------------------------------------------------------------------

spiral = pd.read_csv('SpiralWithCluster.csv', delimiter = ",", usecols = ["x", "y", "SpectralCluster"])
x_train = spiral[['x', 'y']]
y_train = spiral["SpectralCluster"]


# ----------------------------------------------------------------------------
#                           Q2.a
# ----------------------------------------------------------------------------

svmmodel = svm.SVC(kernel = "linear", decision_function_shape = "ovr", random_state = 20210325)
thisfit = svmmodel.fit(x_train, y_train)
y_pred = thisfit.predict(x_train)
spiral['Predicted'] = y_pred
print("Intercept:", thisfit.intercept_)
print("Co-efficient:", thisfit.coef_)
print('-' * 40, end='\n')


# ----------------------------------------------------------------------------
#                           Q2.b
# ----------------------------------------------------------------------------

misclass  = 1 - metrics.accuracy_score(y_train, y_pred)
print('Misclassification Rate:', misclass)
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.c
# ----------------------------------------------------------------------------
inter = thisfit.intercept_
co1 = thisfit.coef_[0][0]
co2 = thisfit.coef_[0][1]
x = np.linspace(-5, 5)
y = (-inter - co1*np.linspace(-5, 5))/co2
reds = spiral[spiral['Predicted'] == 0]
plt.scatter(reds['x'], reds['y'], c="red", label = 0)
blues = spiral[spiral['Predicted'] == 1]
plt.scatter(blues['x'], blues['y'], c = "blue", label = 1)
plt.plot(x, y, color = 'black', linestyle = 'dotted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM Scatter Polot Predicted Class with hyperplane')
plt.grid(True)
plt.legend(title = "Predicted Class")
plt.show()
print('-' * 40, end='\n')

# ----------------------------------------------------------------------------
#                           Q2.d
# ----------------------------------------------------------------------------
spiral['radius'] = np.sqrt(spiral['x']**2 + spiral['y']**2)
spiral['theta'] = np.arctan2(spiral['y'], spiral['x'])
def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)
spiral['theta'] = spiral['theta'].apply(customArcTan)
reds = spiral[spiral['SpectralCluster'] == 0]
plt.scatter(reds['radius'], reds['theta'], c="red", label = 0)
blues = spiral[spiral['SpectralCluster'] == 1]
plt.scatter(blues['radius'], blues['theta'], c = "blue", label = 1)
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM Scatter Plot for Polar Coordinates')
plt.grid(True)
plt.legend(title = "Predicted Class")
plt.show()
print('-' * 40, end='\n')


# ----------------------------------------------------------------------------
#                           Q2.e
# ----------------------------------------------------------------------------
spiral['group'] = 3
spiral.loc[spiral['radius'] <= 2.5, 'group'] = 2
spiral.loc[(spiral['radius'] < 3) & (spiral['theta'] >= 2), 'group'] = 2
spiral.loc[(spiral['radius'] < 3.5) & (spiral['theta'] >= 3), 'group'] = 2
spiral.loc[(spiral['radius'] < 4) & (spiral['theta'] >= 4), 'group'] = 2
spiral.loc[(spiral['radius'] < 2) & (spiral['theta'] >= 3), 'group'] = 1
spiral.loc[(spiral['radius'] < 2.5) & (spiral['theta'] >= 4), 'group'] = 1
spiral.loc[(spiral['radius'] < 3) & (spiral['theta'] >= 5), 'group'] = 1
spiral.loc[(spiral['radius'] < 1.5) & (spiral['theta'] >= 6), 'group'] = 0
reds = spiral[spiral['group'] == 0]
plt.scatter(reds['radius'], reds['theta'], c="red", label = 0)
blues = spiral[spiral['group'] == 1]
plt.scatter(blues['radius'], blues['theta'], c = "blue", label = 1)
green = spiral[spiral['group'] == 2]
plt.scatter(green['radius'], green['theta'], c = "green", label = 2)
black = spiral[spiral['group'] == 3]
plt.scatter(black['radius'], black['theta'], c = "black", label = 3)
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM Scatter Plot for Polar Coordinates having 4 Groups')
plt.grid(True)
plt.legend(title = "Predicted Class", loc='lower right')
plt.show()
print('-' * 40, end='\n')


# ----------------------------------------------------------------------------
#                           Q2.f
# ----------------------------------------------------------------------------
eqs = []
for gr in [(0,1), (1,2), (2,3)]:
    temp_spiral = spiral[spiral['group'].isin(gr)]
    tx_train = temp_spiral[['radius', 'theta']]
    ty_train = temp_spiral["group"]
    svmmodel = svm.SVC(kernel = "linear", 
                       decision_function_shape = "ovr",
                       random_state = 20200408)
    thisfit = svmmodel.fit(tx_train, ty_train)
    y_pred = thisfit.predict(tx_train)
    
    print(f"SVM {gr}")
    print("Intercept:", thisfit.intercept_)
    print("Co-efficient:", thisfit.coef_)
    inter = thisfit.intercept_
    co1 = thisfit.coef_[0][0]
    co2 = thisfit.coef_[0][1]
    x = np.linspace(0, 4.5)
    y = (-inter - co1*x)/co2
    eqs.append((x, y))
    print('*' * 10)
print('-' * 40, end='\n')


# ----------------------------------------------------------------------------
#                           Q2.g
# ----------------------------------------------------------------------------
reds = spiral[spiral['group'] == 0]
plt.scatter(reds['radius'], reds['theta'], c="red", label = 0)
blues = spiral[spiral['group'] == 1]
plt.scatter(blues['radius'], blues['theta'], c = "blue", label = 1)
green = spiral[spiral['group'] == 2]
plt.scatter(green['radius'], green['theta'], c = "green", label = 2)
black = spiral[spiral['group'] == 3]
plt.scatter(black['radius'], black['theta'], c = "black", label = 3)
for eq in eqs:
    plt.plot(eq[0], eq[1], 
         color = 'black', linestyle = 'dotted')
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.title('SVM Scatter Plot for Polar Coordinates with Hyperplanes having 4 Groups')
plt.grid(True)
plt.legend(title = "Pedicted Class")
plt.show()
print('-' * 40, end='\n')


# ----------------------------------------------------------------------------
#                           Q2.h
# ----------------------------------------------------------------------------
ceqs = []
for eq in eqs:
    ceqs.append((eq[0]*np.cos(eq[1]), eq[0]*np.sin(eq[1])))
reds = spiral[spiral['SpectralCluster'] == 0]
plt.scatter(reds['x'], reds['y'], c="red", label = 0)
blues = spiral[spiral['SpectralCluster'] == 1]
plt.scatter(blues['x'], blues['y'], c = "blue", label = 1)
count = 0
for eq in ceqs:
    if count == 0:
        clr = 'green'
    else:
        clr = 'black'
    plt.plot(eq[0], eq[1],color = clr, linestyle = 'dotted')
    count += 1
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM Scatter Plot for Cartesian Coordinates with Hyperplanes')
plt.grid(True)
plt.legend(title = "Predicted Class")
plt.show()
print('-' * 40, end='\n')