import numpy
import statsmodels.api as smodel

# Training partition
xTrain = numpy.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])
yTrain = numpy.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1])

# Build the logistic model
xp1Train = smodel.add_constant(xTrain, prepend = True)
_objMNLogit = smodel.MNLogit(yTrain, xp1Train)

mnlogitFit = _objMNLogit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
print(mnlogitFit.summary())

# Test partition
xTest = numpy.array([0,1,2,3,4])
yTest = numpy.array([1,0,1,0,1])

# Score the logistic model on the Test partition
xp1Test = smodel.add_constant(xTest, prepend = True)
yPredProb = mnlogitFit.predict(xp1Test)

# Calculate the Misclassification Rate
yPredCat = numpy.where(yPredProb[:,1] >= 0.3, 1, 0)
yMCE = numpy.mean(numpy.where(yPredCat == yTest, 0, 1))
print("Logistic Misclassification Rate on Testing Partition = %10.7f" % (yMCE))
