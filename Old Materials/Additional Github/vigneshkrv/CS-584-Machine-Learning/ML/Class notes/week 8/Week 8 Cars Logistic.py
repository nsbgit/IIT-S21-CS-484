import numpy
import pandas
import sympy

import sklearn.metrics as metrics
import statsmodels.api as stats

# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pandas.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100

   print("Frequency Table: \n")
   print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

   return

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

cars = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\cars.csv',
                       delimiter=',',
                       usecols = ['Origin', 'DriveTrain', 'EngineSize', 'Horsepower', 'Length', 'Weight'])

cars = cars.dropna()

TargetPercentByNominal(cars['Origin'], cars['DriveTrain'])

nObs = cars.shape[0]

# Specify Origin as a categorical variable
Origin = cars['Origin'].astype('category')
y = Origin
y_category = y.cat.categories

# Intercept + DriveTrain
DriveTrain = cars[['DriveTrain']].astype('category')
designX = pandas.get_dummies(DriveTrain)
designX = stats.add_constant(designX, prepend=True)

LLK_1, DF_1, fullParams_1 = build_mnlogit (designX, y, debug = 'Y')

# Model is Intercept + DriveTrain + EngineSize + Horsepower + Length + Weight
DriveTrain = cars[['DriveTrain']].astype('category')
designX = pandas.get_dummies(DriveTrain)
designX = designX.join(cars[['EngineSize', 'Horsepower', 'Length', 'Weight']])
designX = stats.add_constant(designX, prepend=True)

LLK_2, DF_2, fullParams_2 = build_mnlogit (designX, y, debug = 'Y')

