# Load the PANDAS library
import pandas

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt

def GetHistCoord (inputVector,          # Input Vector
                  binWidth,             # Bin Width
                  binBoundaryMinimum,   # The minimum bin boundary value
                  binBoundaryMaximum):  # The maximum bin boundary value

    nRows = inputVector.shape[0]
    histCoord = pandas.DataFrame(columns=['MidPoint','Density'])
    midPoint = binBoundaryMinimum + binWidth / 2
    iCoord = 0
    while (midPoint < binBoundaryMaximum):
        u = (inputVector - midPoint) / binWidth
        hCoord = (u[(-0.5 < u) & (u <= 0.5)].count()) / (nRows * binWidth)
        histCoord.loc[iCoord] = [midPoint, hCoord[0]]
        iCoord += 1
        midPoint += binWidth
    return(histCoord)

# Read only the column X from the NormalSample.csv file
vectorX = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\NormalSample.csv',
                          delimiter=',', usecols=["x"])

vectorX.describe()

histogramCoordinate = GetHistCoord (vectorX, 0.1, 45, 55)
print(histogramCoordinate)
plt.figure(figsize=(6,4))
plt.step(histogramCoordinate['MidPoint'], histogramCoordinate['Density'], where = 'mid', label = 'h = 0.5')    
plt.legend()
plt.grid(True)
plt.xticks(np.arange(45,55, 1))
plt.show()

histogramCoordinate = GetHistCoord (vectorX, 0.5, 45, 55)
print(histogramCoordinate)
plt.figure(figsize=(6,4))
plt.step(histogramCoordinate['MidPoint'], histogramCoordinate['Density'], where = 'mid', label = 'h = 0.5')    
plt.legend()
plt.grid(True)
plt.xticks(np.arange(45,55, 1))
plt.show()

histogramCoordinate = GetHistCoord (vectorX, 1, 45, 55)
print(histogramCoordinate)
plt.figure(figsize=(6,4))
plt.step(histogramCoordinate['MidPoint'], histogramCoordinate['Density'], where = 'mid', label = 'h = 1')  
plt.legend()
plt.grid(True)
plt.xticks(np.arange(45,55, 1))
plt.show()

histogramCoordinate = GetHistCoord (vectorX, 2, 45, 55)
print(histogramCoordinate)
plt.figure(figsize=(6,4))
plt.step(histogramCoordinate['MidPoint'], histogramCoordinate['Density'], where = 'mid', label = 'h = 2')    

plt.legend()
plt.grid(True)
plt.xticks(np.arange(45,55, 1))
plt.show()