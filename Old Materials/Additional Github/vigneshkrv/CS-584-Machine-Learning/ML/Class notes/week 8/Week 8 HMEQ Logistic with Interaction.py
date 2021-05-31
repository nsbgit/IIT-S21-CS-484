import numpy
import pandas
import scipy
import sympy 

import statsmodels.api as stats

# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

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

hmeq = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',', usecols = ['BAD', 'JOB', 'REASON'])

hmeq = hmeq.dropna()

# Specify Origin as a categorical variable
y = hmeq['BAD'].astype('category')

# Specify JOB and REASON as categorical variables
xR = pandas.get_dummies(hmeq[['REASON']].astype('category'))
xJ = pandas.get_dummies(hmeq[['JOB']].astype('category'))

# Intercept only model
designX = pandas.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')

# Intercept + REASON
designX = stats.add_constant(xR, prepend=True)
LLK_1R, DF_1R, fullParams_1R = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1R - LLK0)
testDF = DF_1R - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + REASON + JOB
designX = xR
designX = designX.join(xJ)
designX = stats.add_constant(designX, prepend=True)
LLK_1R_1J, DF_1R_1J, fullParams_1R_1J = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1R_1J - LLK_1R)
testDF = DF_1R_1J - DF_1R
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)

# Intercept + REASON + JOB + REASON * JOB
designX = xR
designX = designX.join(xJ)

# Create the columns for the JOB * REASON interaction effect
xRJ = create_interaction(xR, xJ)
designX = designX.join(xRJ)

designX = stats.add_constant(designX, prepend=True)
LLK_2RJ, DF_2RJ, fullParams_2RJ = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_2RJ - LLK_1R_1J)
testDF = DF_2RJ - DF_1R_1J
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)