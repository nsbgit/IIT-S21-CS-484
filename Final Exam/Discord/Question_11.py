import scipy
import numpy as np
import pandas as pd
import statsmodels.api as stats

def SWEEPOperator(pDim, inputM, tol):
    # pDim: dimension of matrix inputM, integer greater than one
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    
    A = np.copy(inputM)
    diagA = np.diagonal(inputM)

    for k in range(pDim):
        Akk = A[k,k]
        if (Akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            ANext = A - np.outer(A[:, k], A[k, :]) / Akk
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
def build_mnlogit(fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = np.transpose(fullX).dot(fullX)
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


credit = pd.read_csv('Q11.csv').dropna()

# create dummy indicators for the categorical input features
# Reorder the categories in ascending order of frequencies, and then Create dummy indicators xCat = []
catName = ['ActiveLifestyle','Gender','MaritalStatus','Retired']
xCat = []
for thisName in catName:
    catFreq = credit[thisName].value_counts() 
    catFreq = catFreq.sort_values(ascending = True) 
    newCat = catFreq.index
    u = credit[thisName].astype('category') 
    xCat.append(pd.get_dummies(u.cat.reorder_categories(newCat))) 

# Reorder the categories in descending order of frequencies of the target field
catFreq = credit['CreditCard'].value_counts()
catFreq = catFreq.sort_values(ascending = False)
newCat = catFreq.index
u = credit['CreditCard'].astype('category')
y = u.cat.reorder_categories(newCat)

chi_dict = {}

# Model 0: Intercept only model 
u = y.isnull()
designX = pd.DataFrame(u.where(u, 1)).rename(columns = {'CreditCard': 'const'}) 
LLK0, DF0, fullParams0, thisFit = build_mnlogit(designX, y)
print('\nModel 0 = Intercept\n')

# Model 1: Intercept + ?
first_model = ['ActiveLifestyle','Gender','MaritalStatus','Retired']
for r in range(4):
    trainData = xCat[r].dropna()
    trainData = stats.add_constant(trainData, prepend = True)
    LLK1, DF1, fullParams1, thisFit = build_mnlogit(trainData, y)
    testDev = 2.0 * (LLK1 - LLK0)
    testDF = DF1 - DF0
    testPValue = scipy.stats.chi2.sf(testDev, testDF)
    if testPValue < 0.01:
        chi_dict[first_model[r]] = testPValue
key_min = min(chi_dict.keys(), key=(lambda k: chi_dict[k]))
print('\nModel 1 = Intercept +', key_min,'\n')


# Model 2 = Intercept + Gender + ?
second_model = [('Gender','ActiveLifestyle'),('Gender','Gender'),('Gender','MaritalStatus'),
                ('Gender','Retired')]
for r in range(4):
    trainData = xCat[1]
    if r!= 1:
        trainData = trainData.join(xCat[r])
        trainData = stats.add_constant(trainData, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit(trainData, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)  
        if testPValue < 0.01:
            chi_dict[second_model[r]] = testPValue
key_min = min(chi_dict.keys(), key=(lambda k: chi_dict[k]))
print('\nModel 2 = Intercept +',' + '.join(''.join(t) for t in key_min),'\n')

# Model 3 = Intercept + Gender + Retired + ?
third_model = [('Gender','Retired','ActiveLifestyle'),0, ('Gender','Retired','MaritalStatus'),0]
for r in range(4):
    if r == 0:
        xCat[0].columns = ['Y','N']
    trainData = xCat[1]
    trainData = trainData.join(xCat[3])
    if r!= 1 and r!=3:
        trainData = trainData.join(xCat[r])
        trainData = stats.add_constant(trainData, prepend = True)
        LLK1, DF1, fullParams1, thisFit = build_mnlogit(trainData, y)
        testDev = 2.0 * (LLK1 - LLK0)
        testDF = DF1 - DF0
        testPValue = scipy.stats.chi2.sf(testDev, testDF)  
        if testPValue < 0.01:
            chi_dict[third_model[r]] = testPValue
key_min = min(chi_dict.keys(), key=(lambda k: chi_dict[k]))
print('\nModel 3 = Intercept +',' + '.join(''.join(t) for t in key_min),'\n')

# Model 4 = Intercept + Gender + Retired + MaritalStatus
fourth_model = [('Gender','Retired','MaritalStatus','ActiveLifestyle')]

xCat[0].columns = ['Y','N']
trainData = xCat[1]
trainData = trainData.join(xCat[3])
trainData = trainData.join(xCat[2])
trainData = trainData.join(xCat[0])
trainData = stats.add_constant(trainData, prepend=True)
LLK1, DF1, fullParams1, thisFit = build_mnlogit(trainData, y)
testDev = 2.0 * (LLK1 - LLK0)
testDF = DF1 - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
if testPValue < 0.01:
    chi_dict[fourth_model[0]] = testPValue
key_min = min(chi_dict.keys(), key=(lambda k: chi_dict[k]))
print('\nModel 4 = Intercept +',' + '.join(''.join(t) for t in key_min),'\n')

print('the last feature that enters into the model is ActiveLifestyle')


