# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_csv('NormalSample.csv')

##############
# Q2.a
##############
print('Q2.a')

# iterate over groups
for g in df['group'].unique():
    # get items in group
    group = df[df['group'].isin([g])]

    # show 5-number summary
    q0 = group['x'].quantile(0.00) # min
    q1 = group['x'].quantile(0.25)
    q2 = group['x'].quantile(0.50) # median
    q3 = group['x'].quantile(0.75)
    q4 = group['x'].quantile(1.00) # max
    print('group %s five-number summary:' % (g))
    print('\tmin =', q0)
    print('\tq1 =', q1)
    print('\tmedian = q2 =', q2)
    print('\tq3 =', q3)
    print('\tmax =', q4)

    # Show 1.5 IQR
    print('\n\t1.5 * IRQ =', 1.5 * (q3 - q1))


##############
# Q2.b
##############
print('\nQ2.b')

# Find outliers for each group
# iterate over groups
for g in df['group'].unique():
    # get items in group
    group = df[df['group'].isin([g])]
        
    # Outliers are more more than 3 standard deviations from the mean
    outliers = group[np.abs(group['x'] - group['x'].mean()) > (3 * group['x'].std())]
    print('\nOutliers in group %s' % g)
    print(outliers)

# Show outliers for entire data set
print('\nOutliers for entire dataset:')
outliers = df[np.abs(df['x'] - df['x'].mean()) > (3 * df['x'].std())]
print(outliers)

# Show boxplot
print('\nSee boxplot window')
df.boxplot(column='x', by='group', vert=False, whis=1.5)
plt.title("Boxplot of x for each group")
plt.xlabel("x")
plt.ylabel("group")
plt.grid(axis="y")
plt.show()
