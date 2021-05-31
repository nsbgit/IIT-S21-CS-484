import pandas as pd
from numpy import percentile
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


dataframe = pd.read_csv("Groceries.csv", delimiter=',')

# Convert the Sale Receipt data to the Item List format
ListItem = dataframe.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format

te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)

# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.01, max_len =2, use_colnames = True)
#print(frequent_itemsets)
# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence",min_threshold=0.1)
#print(assoc_rules)

# Question 1
#Q.1 a.(5 points) Create a data frame that contains the number of unique items in each customer’s market basket. Draw a histogram of the number of unique items.  What are the 25th, 50th, and the 75th percentiles of the histogram?
print("Question 1")
Listitems_count = dataframe.groupby(['Customer'])['Item'].count()
print("Distinct Items of Customers: \n", Listitems_count)
Listitems_count = Listitems_count.sort_values()
#Listitems_count.to_excel('a1.xlsx')

quartiles = percentile(Listitems_count, [25, 50, 75])
print('25th Percentile : ', quartiles[0])
print('50th Percentile : ', quartiles[1])
print('75th Percentile : ', quartiles[2])

plt.hist(Listitems_count)
plt.axvline(quartiles[0], color='blue', linewidth=1, alpha=1)
plt.axvline(quartiles[1], color='yellow', linewidth=1, alpha=1)
plt.axvline(quartiles[2], color='blue', linewidth=1, alpha=1)
plt.title('Histogram : Number of Unique items')
plt.xlabel('Number of Items')
plt.ylabel('Frequency')
plt.show()

# Q1. b. (10 points) We are only interested in the k-itemsets that can be found in the market baskets of at least seventy five (75) customers.  How many itemsets can we find?  Also, what is the largest k value among our itemsets?
print("\nQuestion 2")
Listitems_count = dataframe.groupby(['Customer'])['Item'].count()
Listitems = dataframe.groupby(['Customer'])['Item'].apply(list).values.tolist()
# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(Listitems).transform(Listitems)
item_indicator = pd.DataFrame(te_ary, columns=te.columns_)
minimum_support = 75 / len(Listitems_count)
# Find the frequent itemsets
frequent_itemsets = apriori(item_indicator, min_support=minimum_support, use_colnames=True, max_len=quartiles[1])
largest_k = len(frequent_itemsets['itemsets'][len(frequent_itemsets) - 1])

print("Frequently bought Itemsets: \n", frequent_itemsets['itemsets'])
print("\nTotal No. of Itemsets: ", frequent_itemsets.shape[0])
print("\nLargest k value: ", largest_k)

# Q1. c (10 points) Find out the association rules whose Confidence metrics are greater than or equal to 1%.  How many association rules can we find?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.  Please do not display those rules in your answer.
print("\nQuestion 3")
# Convert the Sale Receipt data to the Item List format
Listitems= dataframe.groupby(['Customer'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format

te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)

# Find the frequent itemsets
frequent_itemsets = apriori(item_indicator, min_support=minimum_support, use_colnames=True, max_len=quartiles[1])
assoc_rules = association_rules(frequent_itemsets, metric = "confidence",min_threshold=0.01)
print("Number of Association Rules Found: ",assoc_rules.shape[0])

# Q1 d. (5 points) Plot the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules you have found in (c).  Please use the Lift metrics to indicate the size of the marker.
print("\nQuestion 4: Plotted Graph")
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], c=assoc_rules['lift'], s=assoc_rules['lift'])
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.grid(True)
cbar = plt.colorbar()
cbar.set_label('Lift', labelpad=+1)
plt.show()

# Q1 e. (5 points) List the rules whose Confidence metrics are greater than or equal to 60%.  Please include their Support and Lift metrics.
print("\nQuestion 5")

assoc_rules = association_rules(frequent_itemsets, metric = "confidence",min_threshold=0.6)
print("Association Rules: \n")
print(assoc_rules.to_string())
