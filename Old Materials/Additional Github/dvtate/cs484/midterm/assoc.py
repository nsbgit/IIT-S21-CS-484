# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load data
df = pd.read_csv('basketball.csv')


# # select customerid, count(item) from (select customerid, item) as t group by customerid
# items_per_customer = df.groupby('Customer')['Item'].nunique()

# print('Five number summary:')
# for i in range(5):
#     print('\tq%d = %d' % (i, items_per_customer.quantile(0.25 * i)))

# items_per_customer.hist()
# plt.title('Histogram of unique items by customer')
# plt.xlabel('Unique items purchased')
# plt.ylabel('Number of customers')
# plt.show()


# # Filter items w/ <75 buyers
# customers_per_item = df.groupby('Item')['Customer'].nunique()
# d = customers_per_item.to_dict()
# pop_items = [k for k in d if d[k] >= 75]
# customers_per_item = customers_per_item[pop_items]

# print('\titem-sets:', len(customers_per_item.keys()))
# print('\tmost popular item:', customers_per_item.quantile(1))

###
# Q1.c
###
print('Q1.c')

# Filter k-itemsets
item_sets = list(map(
    lambda l: list(filter(lambda i: i in df['Item'], l)),
    df.groupby('Customer')['Item'].unique().to_dict().values()))

# Use apriori algorithm
from apyori import apriori
association_rules = list(filter(
    lambda r: len(r.items) == 2,
    list(apriori(
        item_sets,
        min_confidence=0.0,
        min_lift=0,
        min_support=0.000000000001,
        max_length=10))))
print((association_rules))

###
# Q1.d
###
print('Q1.d')

# Alternatively you could find max/min of the ordered stats
x = list(map(lambda r: r.ordered_statistics[0].confidence, association_rules))
y = list(map(lambda r: r.support, association_rules))
s = list(map(lambda r: r.ordered_statistics[0].lift, association_rules))

plt.scatter(x, y, s=s)
plt.xlabel('confidence')
plt.ylabel('support')
plt.show()

###
# Q1.e
###
print('Q1.e')

association_rules = list(apriori(
    item_sets,
    min_confidence=0.6,
    min_lift=0,
    min_support=0.0000001,
    max_length=2))#min_support=0.001))
for r in association_rules:
    print(r)

