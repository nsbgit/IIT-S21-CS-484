# Load the PANDAS library
import pandas
ImgStore = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\Imaginary_Store.csv',
                       delimiter=',')

# Examine a portion of the data frame
print(ImgStore)

# Create frequency 
nItemPerCustomer = ImgStore.groupby(['Customer'])['Item'].count()

freqTable = pandas.value_counts(nItemPerCustomer).reset_index()
freqTable.columns = ['Item', 'Frequency']
freqTable = freqTable.sort_values(by = ['Item'])
print(freqTable)
nItemPerCustomer.describe()

# Convert the Sale Receipt data to the Item List format
ListItem = ImgStore.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.01, max_len = 7, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.5)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.1, max_len = 7, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.8)

assoc_rules['lift'].describe()

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()


# Show rules that have the 'CEREAL' consquent
import numpy as np
Cereal_Consequent_Rule = assoc_rules[np.isin(assoc_rules["consequents"].values, {"Cereal"})]