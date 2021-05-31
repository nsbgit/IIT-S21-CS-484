import pandas as pd
from scipy.stats import iqr
from math import floor, ceil
import matplotlib.pyplot as plt

df = pd.read_csv('NormalSample.csv', index_col='i')

total_elements = len(df.index)

x_list = list(df['x'])

max_x, min_x = max(x_list), min(x_list)

h = (2 * iqr(x_list)) / (total_elements ** (1 / 3))


def w(u):
    if((u <= 1 / 2) and (u > -1 / 2)):
        return 1
    else:
        return 0

# Question 1.a
print("According to Izenman (1991) method, what is the recommended bin-width for the histogram of x?")
print("Recommended bin-width according to Izenman method: ", h)
print()

# Question 1.b
print("What are the minimum and the maximum values of the field x?")
print("Minimum Value: ", min_x)
print("Maximum Value: ", max_x)
print()

# Question 1.c
print("Let a be the largest integer less than the minimum value of the field x, and b be the smallest integer greater than the maximum value of the field x.  What are the values of a and b?")
a = floor(min_x)
b = ceil(max_x)
print("Value of a: ", a)
print("Value of b: ", b)
print()

# Question 1.d,e,f,g


def density_estimate(h, a=floor(min_x), b=ceil(max_x)):

    midpoint_list = [a + (h/2)]
    i = 0

    while((midpoint_list[i]+(h/2)) < b):
        midpoint_list.append(midpoint_list[i]+h)
        i = i + 1
    w_list = []

    for i in range(len(midpoint_list)):
        u_list = []
        for j in range(total_elements):
            u_list.append(w((x_list[j] - midpoint_list[i]) / h))
        w_list.append((sum(u_list)) / (h * total_elements))

    coordinate_list = [(midpoint_list[i],w_list[i]) for i in range(len(midpoint_list))]

    print(coordinate_list)

    plt.hist(x_list, bins=int(10/h),density=True)
    plt.title("Histogram with h = {}".format(h))
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.show()


print("Use h = 0.25, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools.")
density_estimate(0.25)
print()
print("Use h = 0.5, minimum = a and maximum = b. List the coordinates of the density estimator.  Paste the histogram drawn using Python or your favorite graphing tools.")
density_estimate(0.5)
print()
print("Use h = 1, minimum = a and maximum = b. List the coordinates of the density estimator. Paste the histogram drawn using Python or your favorite graphing tools.")
density_estimate(1)
print()
print("Use h = 2, minimum = a and maximum = b. List the coordinates of the density estimator. Paste the histogram drawn using Python or your favorite graphing tools.")
density_estimate(2)
