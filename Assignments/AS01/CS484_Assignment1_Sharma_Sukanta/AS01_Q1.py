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


def printAssignmentDetails():
    print("""Name: Sukanta Sharma\nCourse: CS 484 Introduction to Machine Learning""")
    
    
def calcCD (Y, delta):
   maxY = np.max(Y)
   minY = np.min(Y)
   meanY = np.mean(Y)

   # Round the mean to integral multiples of delta
   middleY = delta * np.round(meanY / delta)

   # Determine the number of bins on both sides of the rounded mean
   nBinRight = np.ceil((maxY - middleY) / delta)
   nBinLeft = np.ceil((middleY - minY) / delta)
   lowY = middleY - nBinLeft * delta

   # Assign observations to bins starting from 0
   m = nBinLeft + nBinRight
   bin_index = 0;
   boundaryY = lowY
   for iBin in np.arange(m):
      boundaryY = boundaryY + delta
      bin_index = np.where(Y > boundaryY, iBin+1, bin_index)

   # Count the number of observations in each bins
   uBin, binFreq = np.unique(bin_index, return_counts = True)

   # Calculate the average frequency
   meanBinFreq = np.sum(binFreq) / m
   ssDevBinFreq = np.sum((Y - meanBinFreq)**2) / m
   CDelta = ((2.0 * meanBinFreq) - ssDevBinFreq) / (delta * delta)
   return(m, middleY, lowY, CDelta, uBin, binFreq)


def checkU (u):
    if u > -1/2 and u <= 1/2:
        return True
    else:
        return False
        
    

printAssignmentDetails()

# **************************  Question 1  ******************************************
normalSampleData = pd.read_csv('NormalSample.csv' ,delimiter=',', usecols=['x'])
normalSampleDataX = normalSampleData['x']

# **************************  Question 1.a  ******************************************
print("\n\n\nQ1.a)	(5 points) Use the Pandas describe() function to find out the count, the mean, the standard deviation, the minimum, the 25th percentile, the median, the 75th percentile, and the maximum.\n")
normalSampleDescribe = normalSampleData.describe()
print(normalSampleDescribe)

# **************************  Question 1.b  ******************************************
print("\n\n\nQ1.b)	(5 points) What is the bin width recommended by the Izenman (1991) method?  Please round your answer to the nearest tenths (i.e., one decimal place).\n")
n = int(normalSampleDescribe['x']['count'])
q3 = normalSampleDescribe['x']['75%']
q1 = normalSampleDescribe['x']['25%']
iqr = q3 - q1
binWidthIzenman = 2 * iqr * (n ** (-1/3))
binWidthIzenman = round(binWidthIzenman, 1)
print(binWidthIzenman)
plt.hist(normalSampleData)
plt.show()

# **************************  Question 1.c  ******************************************
print("\n\n\nQ1.c)	(10 points) Use the Shimazaki and Shinomoto (2007) method and try d = 0.1, 0.2, 0.5, 1.0, 2.0, and 5.0.  What is the recommended bin width?  You need to show your calculations to receive full credit.\n")
result = pd.DataFrame()
deltaList = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
# binMid = np.zero()

for d in deltaList:
    nBin, middleY, lowY, CDelta, uBin, binFreq = calcCD(normalSampleDataX,d)
    highY = lowY + nBin * d
    result = result.append([[d, CDelta, lowY, middleY, highY, nBin, uBin, binFreq]], ignore_index = True)
    
    binMid = lowY + 0.5 * d + np.arange(nBin) * d
    plt.hist(normalSampleDataX, bins = binMid, align='mid')
    plt.title('Delta = ' + str(d))
    plt.ylabel('Number of Observations')
    plt.grid(axis = 'y')
    plt.show()
    
result = result.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin', 6:'uBin', 7:'binFreq'})
# print(result)
sortedResult = result.sort_values(by=['C(Delta)']).reset_index(drop=True)
print(sortedResult)
recommendedBinWidthSSM = sortedResult['Delta'][0]
print("\nTherefore, Recomended bin-width is, d = {0}\n".format(recommendedBinWidthSSM))
fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot')
ax1.boxplot(normalSampleDataX, labels = ['X'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()  

# **************************  Question 1.d  ******************************************
print("\n\n\nd)Q1.d)	(5 points) Based on your recommended bin width answer in (c), list the mid-points and the estimated density function values.  Draw the density estimator as a vertical bar chart using the matplotlib.  You need to properly label the graph to receive full credit.\n")
lowY = sortedResult['Low Y'][0]
# recommendedBinWidthSSM = sortedResult['Low Y'][0]
d = sortedResult['Delta'][0]
nBin = sortedResult['N Bin'][0]
binFreq = sortedResult['binFreq'][0]
N = len(normalSampleDataX)
depth = np.arange(nBin) * d
value_add = lowY + 0.5 * d
binMid = value_add + depth
# binMid = lowY + 0.5 * d + np.arange(nBin) * d
p = []
for m_i in binMid:
    u = (normalSampleDataX - m_i) / d
    w = np.where(np.logical_and(u > -1/2,u<= 1/2), 1, 0)
    sum_w = np.sum(w)
    p_i = sum_w / (N * d)
    p.append(p_i)
midPointVSDensity = pd.DataFrame(
    {'Mid-Points':binMid, 
        'Estimated Density Function Values':p})
print("")
print(midPointVSDensity)
print("")   
# Create Vertical Bar Chart
plt.bar(binMid, p, width=0.8, align='center', label="Bin-width={0}".format(recommendedBinWidthSSM))
plt.legend()
x_tick = np.arange(lowY, max(binMid) + d, 1)
y_tick = np.arange(0, max(p) + 0.025, 0.025)
plt.xticks(x_tick)
plt.yticks(y_tick)
plt.xlabel("Mid-points")
plt.ylabel("Estimated Density Function Values")
plt.title("Mid-points v/s Estimated Density Function Values")


    


# **************************  Question 1  ******************************************
# **************************  Question 1  ******************************************
# **************************  Question 1  ******************************************
# **************************  Question 1  ******************************************
# **************************  Question 1  ******************************************
# **************************  Question 1  ******************************************
# **************************  Question 1  ******************************************
    
    
    
    