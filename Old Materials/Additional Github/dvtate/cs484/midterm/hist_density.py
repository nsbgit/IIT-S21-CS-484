# imports
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import math

# Load data
data = pd.read_csv('2020S2MT_Q5.csv')

##############
# Q1.a
##############
xs = data['x']
a = math.floor(min(xs))
b = math.ceil(max(xs))
print('\nQ1.a')
print('\tmin x: ', min(xs))
print('\t therefore, a = ', a)

print('\tmax x: ', max(xs))
print('\t therefore, b = ', b)

##############
# Q1.b-e
##############

'''
Show density estimator summary and draw a histogram
'''
def show_density_summary(bin_width, xs = xs, a = a, b = b):
    # For each bin
    bin_start = a
    while bin_start < b:
        # Get items in the bin
        entries = filter(lambda x:
            x > bin_start and x <= bin_start + bin_width, xs)
        num_entries = len(tuple(entries))

        # Calculate density
        density = num_entries / (bin_width * len(xs))

        # Display list of densities
        print('\t(%s, %s] : %s occurances; density = %s' % (
            bin_start, bin_start + bin_width, num_entries, density))

        # Continue to next bin
        bin_start += bin_width

    # Show Histogram
    data.hist(column='x', bins=int((b - a) / bin_width))
    plt.title("Histogram of X when h=%s" % (bin_width))
    plt.xlabel("x")
    plt.ylabel("Number of Observations")
    plt.xticks(np.arange(a, b, step=bin_width))
    plt.grid(axis="x")
    plt.show()

show_density_summary(5)


##############
# Q1.f
##############
print('\nQ1.f')

# Print Interquartile-range
q1 = data.quantile(0.25)['x']
q3 = data.quantile(0.75)['x']
iqr = q3 - q1
print('\tinterquartile-range:', iqr)

ideal_h = 2 * iqr * len(xs) ** (-1 / 3)
print('\tIdeal bin-width: h =', ideal_h)
