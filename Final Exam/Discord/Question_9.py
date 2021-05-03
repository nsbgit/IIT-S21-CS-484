import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


Y = ['Non-Event']*14 + ['Event']*6
Y = np.array(Y)

predProbY = np.array([0.0814, 0.1197, 0.1969, 0.3505, 0.3878, 0.3940, 
                      0.4828, 0.4889, 0.5587, 0.5614, 0.6175, 0.6342, 
                      0.6527, 0.6668, 0.4974, 0.6732, 0.6744, 0.6836, 
                      0.7475, 0.7828])

# Generate the coordinates for the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y, predProbY,
                                         pos_label = 'Event')

# Add two dummy coordinates
OneMinusSpecificity = np.append([0], fpr)
Sensitivity = np.append([0], tpr)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])

# Draw the Kolmogorov Smirnov curve
cutoff = np.where(thresholds > 1.0, np.nan, 
                     thresholds)
plt.plot(cutoff, tpr, marker = 'o', 
         label = 'True Positive',
         color = 'blue', linestyle = 'solid')
plt.plot(cutoff, fpr, marker = 'o',
         label = 'False Positive',
         color = 'red', linestyle = 'solid')
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True)
plt.show()

print('thresholds = ', thresholds)
print('Prob. threshold that yields the highest Kolmogorov–Smirnov statistic is p = ',
      thresholds[2])
