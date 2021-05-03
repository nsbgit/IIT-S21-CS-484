import numpy as np
import pandas as pd

def CramerV(xCat, yCat):
    # Generate the crosstabulation
    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    xNCat = obsCount.shape[0]
    yNCat = obsCount.shape[1]
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))
    # Calculate the Chi-Square statistics
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    Cramer_V = np.sqrt(chiSqStat/((nTotal)*min(xNCat - 1, yNCat - 1)))
    return Cramer_V
# create our data
# we first store all data in python dictionary
# it works as follws, we create an empty dict with row and coln as keys and empty lists as values
# we then apply extend (instead of using for loop!). If we add up all the values in Row A, that
# is the total number of times A elements in the dataset. It's crossing with 1,2,3, and 4 is given
# by the cross tablulation. 
# you extend the row based on the total sum of that particular row (this denotes it's total
# number of appearances in the dataset) and then for each of the cross value, you add that many
# values in the dictionary list for it. So if crosstab A-1 = 4340, we create 4340 rows for A with
# a column value of 1.
# this works for rest as well, we get a very large dict at end
d = {"Row": [], "Column": []}
d['Row'].extend(['A']*(4340 + 5403 + 2456 + 353))
d['Column'].extend([1]*4340)
d['Column'].extend([2]*5403)
d['Column'].extend([3]*2456)
d['Column'].extend([4]*353)
d['Row'].extend(['B']*(8095 + 16156 + 10798 + 2371))
d['Column'].extend([1]*8095)
d['Column'].extend([2]*16156)
d['Column'].extend([3]*10798)
d['Column'].extend([4]*2371)
d['Row'].extend(['C']*(4761 + 14154 + 14103 + 4597))
d['Column'].extend([1]*4761)
d['Column'].extend([2]*14154)
d['Column'].extend([3]*14103)
d['Column'].extend([4]*4597)
d['Row'].extend(['D']*(813 + 3636 + 5307 + 2657))
d['Column'].extend([1]*813)
d['Column'].extend([2]*3636)
d['Column'].extend([3]*5307)
d['Column'].extend([4]*2657)
# transform dict into pandas dataframe so we can pass columns into our Pearson function!
data = pd.DataFrame(data=d)
# get Cramer's V
print('Cramers V = ',CramerV(data['Row'], data['Column']))
