import numpy
import pandas
import matplotlib.pyplot as plt
import math

Groceries = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\Groceries.csv',
                            delimiter=',', usecols=['Customer', 'Item'])

# Convert the Groceries data to the Item List format
ListItem = Groceries.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Part (a)
nCustomer = len(ListItem)
print("Number of Customers = ", nCustomer)

# Part (b)
print("Number of Unique Items = ", len(set(Groceries['Item'])))

# Part (c)
nItem = numpy.zeros(nCustomer)
for i in range(nCustomer):
    nItem[i] = len(ListItem[i])

p25 = numpy.percentile(nItem, 25)
p50 = numpy.percentile(nItem, 50)
p75 = numpy.percentile(nItem, 75)

print("Number of Items Ever Purhcased by a Customer:/n")
print("Minimum = ", numpy.min(nItem))
print("25th Percentile = ", p25)
print("50th Percentile = ", p50)
print("75th Percentile = ", p75)
print("Maximum = ", numpy.max(nItem))

plt.hist(nItem, bins = 32)
plt.xticks(numpy.arange(0, 32, step = 2))
plt.grid(True, axis = 'both')
plt.xlabel("Number of Items Per Customer")
plt.ylabel("Frequency")
plt.show()

# Part (d)
# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Minimum support is 75 / 9835
frequent_itemsets = apriori(ItemIndicator, min_support = (75/9835), max_len = 32, use_colnames = True)

# Part (e)
# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)

# Part (f)
plt.scatter(x=assoc_rules['confidence'], y=assoc_rules['support'], s=20*assoc_rules['lift'])
plt.grid(True, axis = 'both')
plt.xlim(0.0, 0.8)
plt.ylim(0.0, 0.1)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

# Part (g)
Over60Rules = assoc_rules[assoc_rules['confidence'] >= 0.6]
print("Association Rules whose Confidence is at least 60%: /n")
print(Over60Rules)
