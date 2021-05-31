# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy
import pandas

rocData = pandas.DataFrame({'CutOff': [-2, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 2], 
                            'Sensitivity': [1.0, 1.0, 1.0, 0.83, 0.67, 0.5, 0.33, 0.33, 0.17, 0],
                            'OneMinusSpecificity': [1.0, 1.0, 0.8, 0.4, 0.4, 0.2, 0.2, 0, 0, 0]})

ax = plt.gca()
ax.set_aspect('equal')
plt.figure(figsize=(6,6))
plt.plot(rocData['OneMinusSpecificity'], rocData['Sensitivity'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.show()