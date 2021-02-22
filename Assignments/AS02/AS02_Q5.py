# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:47:59 2021

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Spring 2021
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#  ------------------ Question 5 ----------------------
df = pd.read_csv('Groceries.csv')


#   ----------------- Question 5.a ----------------
dataItems = df.groupby(['Customer'])['Item'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(dataItems).transform(dataItems)
itemIndicators = pd.DataFrame(te_ary, columns = te.columns_)
frequent_item_sets = apriori(itemIndicators, min_support = 75 / len(dataItems), use_colnames = True, max_len = None)
totalItemCount = len(frequent_item_sets)
print("{0} itemsets can we find in total.".format(totalItemCount))
largestK = len(frequent_item_sets['itemsets'][totalItemCount - 1])
print("the largest k value among our itemsets is {0}".format(largestK))


# ---------------------- Question 5.b ---------------------------------
# Find the frequent itemsets
frequent_itemsets_b = apriori(itemIndicators, min_support = 75 / len(dataItems), max_len = largestK, use_colnames = True)
assoc_rules_b = association_rules(frequent_itemsets_b, metric = "confidence", min_threshold = 0.01)
numberOfRules = len(assoc_rules_b['lift'])
print("We can found {0} rules".format(numberOfRules))


# --------------- Question 5.c ---------------------------
plt.scatter(assoc_rules_b['confidence'], assoc_rules_b['support'], c = assoc_rules_b['lift'], s = assoc_rules_b['lift'])
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.title("Support vs Confidence")
color_bar = plt.colorbar()
color_bar.set_label("Lift")
plt.show()

# ----------------- Question 5.d -----------------------------
ass_rules_e = association_rules(frequent_item_sets, metric = "confidence", min_threshold = 0.6)
print(ass_rules_e)
