# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:48:44 2021

@author: Sukanta Sharma
Name: Sukanta Sharma
Student Id: A20472623
Course: CS 484 - Introduction to Machine Learning
Semester:  Splring 2021
"""

import pandas
import numpy
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ------------------------ Question 2 ------------------------------------------------
ToyAssoc = pandas.read_csv('q2data.csv', delimiter=',')

# Convert the Sale Receipt data to the Item List format
ListItem = ToyAssoc.groupby(['Friend'])['Item'].apply(list).values.tolist()

# Convert the Item List format to the Item Indicator format

te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pandas.DataFrame(te_ary, columns=te.columns_)



# Find the frequent itemsets
frequent_itemsets = apriori(ItemIndicator, min_support = 0.3, max_len = 3, use_colnames = True)

# Discover the association rules
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.5)

# Show rules that have the 'CEREAL' consquent

Cereal_Consequent_Rule = assoc_rules[numpy.isin(assoc_rules["consequents"].values, {"Soda"})]

# Show rules that have the 'Oranges' antecedent
antecedent = assoc_rules["antecedents"]
selectAntecedent = numpy.ones((assoc_rules.shape[0], 1), dtype=bool)

i = 0
for f in antecedent:
    # selectAntecedent[i,0] = "Oranges" in f
    selectAntecedent[i,0] = f.issubset(set(['Cheese', 'Wings']))
    i = i + 1
  
Orange_Antecedent_Rule = assoc_rules[selectAntecedent]



# ------------------ Question 4 --------------------------------------
clusters = [[-2, -1, 1, 2, 3], [4, 5, 7, 8]]

# ------------------------- Question 4. a ----------------------------
observation = -1
a_ij = sum(abs(x - observation) for x in clusters[0]) / (len(clusters[0]) - 1)
b_ij = sum(abs(x - observation) for x in clusters[1] if x != observation) / len(clusters[0])
s = (b_ij - a_ij) / max(a_ij, b_ij)
print("Silhouette Width of the observation is : {0}".format(s))


# ---------------------- Question 4.b --------------------------  
s_0 = sum(abs(x - numpy.mean(clusters[0])) for x in clusters[0]) / len(clusters[0])
s_1 = sum(abs(x - numpy.mean(clusters[1])) for x in clusters[1]) / len(clusters[1])
print("Davies-Bouldin value of Cluster 0  : {0}".format(s_0))
print("Davies-Bouldin value of Cluster 1  : {0}".format(s_1))


# -------------------- Question 4.c -------------------------
m_01 = abs(numpy.mean(clusters[0]) - numpy.mean(clusters[1]))
r_01 = (s_0 + s_1) / m_01
db = r_01 / len(clusters)
print("\n\n\nDavies-Bouldin value of Cluster 0  : {0}".format(db))