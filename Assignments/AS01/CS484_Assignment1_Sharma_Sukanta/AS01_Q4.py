# -*- coding: utf-8 -*-
"""

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Spring 2021
"""

# Load the necessary libraries
import graphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree


def CosineD (x, y):
    normX = np.sqrt(np.dot(x, x))
    normY = np.sqrt(np.dot(y, y))
    if normX > 0.0 and normY > 0.0:
        outDistance = 1.0 - np.dot(x, y) / normX / normY
    else:
        outDistance = np.nan
    return outDistance


# **************************  Question 4  ******************************************
airport = pd.read_csv('Airport.csv',delimiter=',')
# Remove all missing observations
# airport = airport.dropna()

# **************************  Question 4.a  ******************************************
# x-axis Airport 3 (y-axis) versus Airport 2 (x-axis).  

xairport = airport['Airport 2']
yairport = airport['Airport 3']
# girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
# boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
# grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]#x
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(xairport, yairport)
# ax.scatter(grades_range, boys_grades, color='b')
ax.set_xlabel('Airport 2')
ax.set_ylabel('Airport 3')
ax.set_title('Airport 3 (y-axis) versus Airport 2 (x-axis)')
plt.show()


# **************************  Question 4.b  ******************************************
# Put the descriptive statistics into another dataframe
airport_descriptive = airport.describe()
a2 = airport['Airport 2'].to_frame()
a3 = airport['Airport 3'].to_frame()
a3 = a3.rename(columns={"Airport 3": "Airport 2"})
# a3 = a3.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin', 6:'uBin', 7:'binFreq'})
freq_data = a2.append(a3)
# freq_data = pd.concat([a2,a3]).drop_duplicates().reset_index(drop=True)
freq_data.groupby('Airport 2').size().plot(kind='barh')
plt.title("frequency table of the airport codes in Airport 2 and Airport 3 combined")
plt.xlabel("Number of Observations")
plt.ylabel("Airports")
plt.grid(axis="x")

plt.show()

a2.groupby('Airport 2').size().plot(kind='barh')
plt.title("frequency table of the airport codes in Airport 2 and Airport 3 combined")
plt.xlabel("Number of Observations")
plt.ylabel("Airports")
plt.grid(axis="x")

plt.show()

# **************************  Question 4.c.i  ******************************************
xdf = pd.read_csv('Airport_FreqData.csv',delimiter=',')
pdf = pd.read_csv('Airport_ProbeData.csv',delimiter=',')
X = xdf.to_numpy()
P = pdf.to_numpy()
xLen = len(X)
pLen = len(P)
cosine_D = np.zeros((xLen, pLen))
for i in range(xLen):
    for j in range(pLen):
        pass
        cosine_D[i, j] = CosineD(X[i, :], P[j, :])

print("\nCosine Distance")
print(cosine_D)

minInColumns = np.amin(cosine_D, axis=0)

minIndex = np.where(cosine_D == np.amin(cosine_D, axis=1))

print("\nMinimum Distance")
print(minInColumns)

print("\nIndex of Minimum Distance")
print(minIndex)

# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
# **************************  Question 4  ******************************************
