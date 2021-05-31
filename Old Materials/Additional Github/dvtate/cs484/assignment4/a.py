
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy
import statsmodels.api as stats

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

def SWEEPOperator (pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []

    A = numpy.copy(inputM)
    diagA = numpy.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / Akk
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:, k] = 0.0 * A[:, k]
            ANext[k, :] = ANext[:, k]
        A = ANext
    return (A, aliasParam, nonAliasParam)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = numpy.transpose(fullX).dot(fullX)
    invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-7)

    # Build a multinomial logistic model
    X = fullX.iloc[:, list(nonAliasParam)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method = 'newton', maxiter = 1000, gtol = 1e-6, full_output = True, disp = True)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    # The number of free parameters
    nYCat = thisFit.J
    thisDF = len(nonAliasParam) * (nYCat - 1)

    # Return model statistics
    return (thisLLK, thisDF, thisParameter, thisFit)

df = pandas.read_csv('./Purchase_Likelihood.csv')
df['group_size_x_homeowner'] = df['group_size'] * df['homeowner']
df['group_size_x_married_couple'] = df['group_size'] * df['homeowner']
df['homeowner_x_married_couple'] = df['homeowner'] * df['married_couple']
training_fields = (
    'group_size',
    'homeowner',
    'married_couple',
    'group_size_x_homeowner',
    'group_size_x_married_couple',
    'homeowner_x_married_couple',
)

fX = df[training_fields]
X = numpy.transpose(fX).dot(fX)
invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = fX, inputM = X, tol = 1e-7)

print(invXtX, aliasParam, nonAliasParam)