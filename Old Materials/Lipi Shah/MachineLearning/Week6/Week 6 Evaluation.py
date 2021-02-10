import matplotlib.pyplot as plt
import numpy
import sklearn.metrics as metrics

Y = numpy.array(['Event',
                 'Non-Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Non-Event',
                 'Event',
                 'Event',
                 'Non-Event'])

nY = Y.shape[0]

predProbY = numpy.array([0.9,0.5,0.3,0.7,0.3,0.8,0.4,0.2,1,0.5,0.3])

# Determine the predicted class of Y
predY = numpy.empty_like(Y)
for i in range(nY):
    if (predProbY[i] > 0.5):
        predY[i] = 'Event'
    else:
        predY[i] = 'Non-Event'

# Calculate the Root Average Squared Error
RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Event'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = numpy.sqrt(RASE/nY)

# Calculate the Root Mean Squared Error
Y_true = 1.0 * numpy.isin(Y, ['Event'])
RMSE = metrics.mean_squared_error(Y_true, predProbY)
RMSE = numpy.sqrt(RMSE)

# For binary y_true, y_score is supposed to be the score of the class with greater label.
AUC = metrics.roc_auc_score(Y_true, predProbY)
accuracy = metrics.accuracy_score(Y, predY)

print('                  Accuracy: {:.13f}' .format(accuracy))
print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
print('          Area Under Curve: {:.13f}' .format(AUC))
print('Root Average Squared Error: {:.13f}' .format(RASE))
print('   Root Mean Squared Error: {:.13f}' .format(RMSE))

# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Event')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
