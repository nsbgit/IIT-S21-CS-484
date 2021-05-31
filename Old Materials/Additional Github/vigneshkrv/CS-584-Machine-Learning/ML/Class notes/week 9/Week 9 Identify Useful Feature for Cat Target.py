import numpy
import pandas
import scipy
import sklearn.ensemble as ensemble
import statsmodels.api as smodel

# Define a function that performs the Chi-square test
def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pandas.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    if (debug == 'Y'):
        print('Observed Count:\n', obsCount)
        print('Column Total:\n', cTotal)
        print('Row Total:\n', rTotal)
        print('Overall Total:\n', nTotal)
        print('Expected Count:\n', expCount)
        print('\n')
       
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = numpy.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)

# Define a function that performs the Deviance test
def DevianceTest (
    xInt,           # input interval feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    y = yCat.astype('category')

    # Model 0 is yCat = Intercept
    X = numpy.where(yCat.notnull(), 1, 0)
    objLogit = smodel.MNLogit(y, X)
    thisFit = objLogit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK0 = objLogit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Log-Likelihood Value =", LLK0)
        print('\n')

    # Model 1 is yCat = Intercept + xInt
    X = smodel.add_constant(xInt, prepend = True)
    objLogit = smodel.MNLogit(y, X)
    thisFit = objLogit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK1 = objLogit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Log-Likelihood Value =", LLK1)

    # Calculate the deviance
    devianceStat = 2.0 * (LLK1 - LLK0)
    devianceDf = (len(y.cat.categories) - 1.0)
    devianceSig = scipy.stats.chi2.sf(devianceStat, devianceDf)

    mcFaddenRSq = 1.0 - (LLK1 / LLK0)

    return(devianceStat, devianceDf, devianceSig, mcFaddenRSq)

# The Home Equity Loan example
catPred = ['REASON', 'JOB', 'DEROG', 'DELINQ', 'NINQ']
intPred = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'CLAGE', 'CLNO', 'DEBTINC']

hmeq = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\hmeq.csv',
                       delimiter=',', usecols = ['BAD']+catPred+intPred)
hmeq = hmeq.dropna()

testResult = pandas.DataFrame(index = catPred + intPred,
                              columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in catPred:
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(hmeq[pred], hmeq['BAD'], debug = 'Y')
    testResult.loc[pred] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]
    
for pred in intPred:
    devianceStat, devianceDf, devianceSig, mcFaddenRSq = DevianceTest(hmeq[pred], hmeq['BAD'], debug = 'Y')
    testResult.loc[pred] = ['Deviance', devianceStat, devianceDf, devianceSig, 'McFadden''s R^2', mcFaddenRSq]

rankSig = testResult.sort_values('Significance', axis = 0, ascending = True)
print(rankSig)

rankAssoc = testResult.sort_values('Measure', axis = 0, ascending = False)
print(rankAssoc)

# Feature Importance from Random Forest
x_category = hmeq[catPred].astype('category')
xCat = pandas.get_dummies(x_category)
xData = xCat.join(hmeq[intPred])

yData = hmeq['BAD'].astype('category')

_objRF = ensemble.RandomForestClassifier(criterion = 'entropy', n_estimators = 1000, max_features = 'sqrt',
                                         max_depth = 10, random_state = 27513, bootstrap = True)
thisRandomForest = _objRF.fit(xData, yData)
thisFeatureImp = thisRandomForest.feature_importances_

xFreq = xData.sum(0)