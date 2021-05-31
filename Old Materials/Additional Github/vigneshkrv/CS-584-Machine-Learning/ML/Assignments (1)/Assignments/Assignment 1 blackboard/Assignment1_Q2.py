# Load the PANDAS library
import pandas
import matplotlib.pyplot as plt

# Read only the column x from the NormalSample.csv file
OnlyX = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\NormalSample.csv',
                        delimiter=',', usecols=["x"])

# Assign a dummy Group value to the OnlyX to indicate that it is a overall
OnlyX["Group"] = "Overall"

# Read both columns x and Group from the NormalSample.csv file
GroupX = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\NormalSample.csv',
                          delimiter=',', usecols=["x","Group"])

# Concatenating both data frames
Q2Data = pandas.concat([GroupX, OnlyX])

# Obtain the five-number summary for the overall and for each Group category
Q2Data.groupby('Group').describe()

# Visualize the boxplot of the X variable
OnlyX.boxplot(column='x', vert=False, figsize=(6,4))

# Visualize the boxplot of the X variable by GROUP
Q2Data.boxplot(column='x', by='Group', vert=False, figsize=(6,4))
