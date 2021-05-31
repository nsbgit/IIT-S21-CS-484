import pandas as pd
from numpy import percentile
import matplotlib.pyplot as plt

df = pd.read_csv('NormalSample.csv', index_col='i')

total_items = len(df.index)

x_list = list(df['x'])
group_list = list(df['group'])
category1_list=[]
category0_list=[]
for i in range(len(df.index)):
    if group_list[i]==1:
        category1_list.append(x_list[i])
    else:
        category0_list.append(x_list[i])

def outlier(x, lw, rw):
    if((x > rw) or (x < lw)):
        return True
    else:
        return False


# Question 1

max_x, min_x = max(x_list), min(x_list)

quartiles = percentile(x_list, [25, 50, 75])

print('Minimum: ', min_x)
print('Q1: ', quartiles[0])
print('Median: ', quartiles[1])
print('Q3: ', quartiles[2])
print('Maximum: ', max_x)

IQR = quartiles[2] - quartiles[0]
lr = 1.5 * IQR
lw = quartiles[0] - lr
rw = quartiles[2] + lr
LW = max(min_x,lw)
RW = min(max_x,rw)
print('Lower whisker value:',LW)
print('Upper whisker value:',RW)

# Question 3

fig = plt.figure()
only_x = fig.add_subplot(111)

plt.title("BoxPlot(x) without Groups")
plt.grid(True)
only_x.boxplot(x_list,vert=False)
only_x.set_yticklabels('x')
plt.show()

#Category 0
# Question 2

max_x0, min_x0 = max(category0_list), min(category0_list)

quartiles0 = percentile(category0_list, [25, 50, 75])
print('-'*50)
print('Group 0')
print('Minimum: ', min_x0)
print('Q1: ', quartiles0[0])
print('Median: ', quartiles0[1])
print('Q3: ', quartiles0[2])
print('Maximum: ', max_x0)

IQR0 = quartiles0[2] - quartiles0[0]
lr0 = 1.5 * IQR0
lw0 = quartiles0[0] - lr0
rw0 = quartiles0[2] + lr0
LW0 = max(min_x0,lw0)
RW0 = min(max_x0,rw0)

print('Lower whisker value for Category 0:',LW0)
print('Upper whisker value for Category 0:',RW0)

#Category 1

max_x1, min_x1 = max(category1_list), min(category1_list)

quartiles1 = percentile(category1_list, [25, 50, 75])
print('-'*50)
print('Group 1')
print('Minimum: ', min_x1)
print('Q1: ', quartiles1[0])
print('Median: ', quartiles1[1])
print('Q3: ', quartiles1[2])
print('Maximum: ', max_x1)

IQR1 = quartiles1[2] - quartiles1[0]
lr1 = 1.5 * IQR1
lw1 = quartiles1[0] - lr1
rw1 = quartiles1[2] + lr1
LW1 = max(min_x1,lw1)
RW1 = min(max_x1,rw1)
print('Lower whisker value for Category 1:',LW1)
print('Upper whisker value for Category 1:',RW1)
print()

# Question 4

fig1 = plt.figure()
x_with01 = fig1.add_subplot(111)

plt.title("Boxplot(x) with Category0 and Category1")
plt.grid(True)
x_with01.boxplot([x_list, category0_list, category1_list], vert=False)
x_with01.set_yticklabels(['X', 'Category0', 'Category1'])
plt.show()

plt.clf()
plt.close()

outliers = [x_list[i] for i in range(len(x_list)) if(outlier(x_list[i], LW, RW))]

outliers_cat_0 = [category0_list[i] for i in range(len(category0_list)) if(outlier(category0_list[i], LW0, RW0))]

outliers_cat_1 = [category1_list[i] for i in range(len(category1_list)) if(outlier(category1_list[i], LW1, RW1))]

print("Outliers for Entire Data: ", outliers)

print("Outliers for Category 0: ", outliers_cat_0)

print("Outliers for Category 1: ", outliers_cat_1)