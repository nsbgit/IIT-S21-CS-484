# A Python program to print all combinations of given length 

import numpy as np
from itertools import combinations

Universal = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

nComb = np.zeros(7)

for i in range(7):
   Size = i + 1
   comb = combinations(Universal, Size)    # Get all permutations of length Size

   print("Combinations of Size ", Size)
   for j in list(comb):
      nComb[i] += 1
      print(j)

nTotalComb = 0
for i in range(7):
   Size = i + 1
   nTotalComb += nComb[i]
   print("Size = ", Size, "Number of Combinations = ", nComb[i])

print("Total Number of Combinations = ", nTotalComb)