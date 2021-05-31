import numpy
import pandas
import scipy
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split

import statsmodels.api as stats

# The SWEEP Operator
def SWEEPOperator (pDim, inputM, tol):
    # pDim: dimension of matrix inputM, positive integer
    # inputM: a square and symmetric matrix, numpy array
    # tol: singularity tolerance, positive real

    aliasParam = []
    nonAliasParam = []
    A = numpy.array(inputM, copy = True, dtype = numpy.float)
    diagA = numpy.diagonal(A)
 
    for k in range(pDim):
        akk = A[k,k]
        if (akk >= (tol * diagA[k])):
            nonAliasParam.append(k)
            for i in range(pDim):
                if (i != k):
                    for j in range(pDim):
                        if (j != k):
                            A[i,j] = A[i,j] - A[i,k] * (A[k,j] / akk)
                            A[j,i] = A[i,j]
                A[i,k] = A[i,k] / akk
                A[k,i] = A[i,k]
            A[k,k] = - 1.0 / akk
        else:
            aliasParam.append(k)
            for i in range(pDim):
                A[i,k] = 0.0
                A[k,i] = 0.0
    return A, aliasParam, nonAliasParam

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y):

    # Find the non-redundant columns in the design matrix fullX
    nFullParam = fullX.shape[1]
    XtX = numpy.transpose(fullX).dot(fullX)
    invXtX, aliasParam, nonAliasParam = SWEEPOperator(pDim = nFullParam, inputM = XtX, tol = 1e-13)

    # Build a multinomial logistic model
    X = fullX.iloc[:, list(nonAliasParam)]
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='ncg', maxiter = 1000, xtol = 1e-8,  
                        full_output = True, disp = True)
    thisParameter = thisFit.params
    thisLLK = thisFit.llf
    
    # The number of free parameters
    y_category = y.cat.categories
    nYCat = len(y_category)
    thisDF = len(nonAliasParam) * (nYCat - 1)

    # Return model statistics
    return (thisLLK, thisDF, thisParameter, thisFit)

inputData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\policy_2001.csv',
                            delimiter=',',
                            usecols = ['CLAIM_FLAG', 'CREDIT_SCORE_BAND', 'BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME'])

# Print number of missing values per variable
print('Number of Missing Values:')
print(pandas.Series.sort_index(inputData.isna().sum()))

# Specify CLAIM_FLAG as a categorical variable
inputData['CLAIM_FLAG'] = inputData['CLAIM_FLAG'].astype('category')
y_category = inputData['CLAIM_FLAG'].cat.categories
nYCat = len(y_category)

# Specify CREDIT_SCORE_BAND as a categorical variable
inputData['CREDIT_SCORE_BAND'] = inputData['CREDIT_SCORE_BAND'].astype('category')

# Create Training and Test partitions
policy_train, policy_test = train_test_split(inputData, test_size = 0.33, random_state = 20201014, stratify = inputData['CLAIM_FLAG'])

nObs_train = policy_train.shape[0]
nObs_test = policy_test.shape[0]

# Build the logistic model
y = policy_train['CLAIM_FLAG']

# Train a Logistic Regression model using the Forward Selection method
devianceTable = pandas.DataFrame()

u = pandas.DataFrame()

# Step 0: Intercept only model
u = y.isnull()
designX = pandas.DataFrame(u.where(u, 1)).rename(columns = {'CLAIM_FLAG': 'const'})
LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)
devianceTable = devianceTable.append([[0, 'Intercept', DF0, LLK0, None, None, None]])

# Consider Model 1 is CLAIM_FLAG = Intercept + <predictor>
predList = ['CREDIT_SCORE_BAND', 'BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']

step = 1.0
for pred in predList:
   step += 0.1
   thisVar = policy_train[pred]
   dType = thisVar.dtypes.name
   if (dType == 'category'):
      designX = pandas.get_dummies(thisVar)
   else:
      designX = thisVar
   designX = stats.add_constant(designX, prepend=True)

   LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
   testDev = 2.0 * (LLK1 - LLK0)
   testDF = DF1 - DF0
   testPValue = scipy.stats.chi2.sf(testDev, testDF)
   devianceTable = devianceTable.append([[step, 'Intercept + ' + pred,
                                          DF1, LLK1, testDev, testDF, testPValue]])

# Step 1: Model is CLAIM_FLAG = Intercept + MVR_PTS
designX = policy_train[['MVR_PTS']]
designX = stats.add_constant(designX, prepend=True)
LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)
devianceTable = devianceTable.append([[1, 'Intercept + MVR_PTS',
                                       DF0, LLK0, None, None, None]])

# Consider Model 2 is CLAIM_FLAG = Intercept + MVR_PTS + <predictor>
predList = ['CREDIT_SCORE_BAND', 'BLUEBOOK_1000', 'CUST_LOYALTY', 'TIF', 'TRAVTIME']

step = 2.0
for pred in predList:
   step += 0.1
   designX = policy_train[['MVR_PTS']]
   thisVar = policy_train[pred]
   dType = thisVar.dtypes.name
   if (dType == 'category'):
      designX = designX.join(pandas.get_dummies(thisVar))
   else:
      designX = designX.join(thisVar)
   designX = stats.add_constant(designX, prepend=True)

   LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
   testDev = 2.0 * (LLK1 - LLK0)
   testDF = DF1 - DF0
   testPValue = scipy.stats.chi2.sf(testDev, testDF)
   devianceTable = devianceTable.append([[step, 'Intercept + MVR_PTS + ' + pred,
                                          DF1, LLK1, testDev, testDF, testPValue]])

# Step 2: Model is CLAIM_FLAG = Intercept + MVR_PTS + BLUEBOOK_1000
designX = policy_train[['MVR_PTS','BLUEBOOK_1000']]
designX = stats.add_constant(designX, prepend=True)
LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)
devianceTable = devianceTable.append([[2, 'Intercept + MVR_PTS + BLUEBOOK_1000',
                                       DF0, LLK0, None, None, None]])

# Consider Model 2 is CLAIM_FLAG = Intercept + MVR_PTS + BLUEBOOK_1000 + <predictor>
predList = ['CREDIT_SCORE_BAND', 'CUST_LOYALTY', 'TIF', 'TRAVTIME']

step = 3.0
for pred in predList:
   step += 0.1
   designX = policy_train[['MVR_PTS','BLUEBOOK_1000']]
   thisVar = policy_train[pred]
   dType = thisVar.dtypes.name
   if (dType == 'category'):
      designX = designX.join(pandas.get_dummies(thisVar))
   else:
      designX = designX.join(thisVar)
   designX = stats.add_constant(designX, prepend=True)

   LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
   testDev = 2.0 * (LLK1 - LLK0)
   testDF = DF1 - DF0
   testPValue = scipy.stats.chi2.sf(testDev, testDF)
   devianceTable = devianceTable.append([[step, 'Intercept + MVR_PTS + BLUEBOOK_1000 + ' + pred,
                                          DF1, LLK1, testDev, testDF, testPValue]])

# Step 3: Model is CLAIM_FLAG = Intercept + MVR_PTS + BLUEBOOK_1000 + TRAVTIME
designX = policy_train[['MVR_PTS','BLUEBOOK_1000','TRAVTIME']]
designX = stats.add_constant(designX, prepend=True)
LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)
devianceTable = devianceTable.append([[3, 'Intercept + MVR_PTS + BLUEBOOK_1000 + TRAVTIME',
                                       DF0, LLK0, None, None, None]])

# Consider Model 2 is CLAIM_FLAG = Intercept + MVR_PTS + BLUEBOOK_1000 + TRAVTIME + <predictor>
predList = ['CREDIT_SCORE_BAND', 'CUST_LOYALTY', 'TIF']

step = 4.0
for pred in predList:
   step += 0.1
   designX = policy_train[['MVR_PTS','BLUEBOOK_1000','TRAVTIME']]
   thisVar = policy_train[pred]
   dType = thisVar.dtypes.name
   if (dType == 'category'):
      designX = designX.join(pandas.get_dummies(thisVar))
   else:
      designX = designX.join(thisVar)
   designX = stats.add_constant(designX, prepend=True)

   LLK1, DF1, fullParams1, thisFit = build_mnlogit (designX, y)
   testDev = 2.0 * (LLK1 - LLK0)
   testDF = DF1 - DF0
   testPValue = scipy.stats.chi2.sf(testDev, testDF)
   devianceTable = devianceTable.append([[step, 'Intercept + MVR_PTS + BLUEBOOK_1000 + TRAVTIME + ' + pred,
                                          DF1, LLK1, testDev, testDF, testPValue]])
   
# Final Model is CLAIM_FLAG = Intercept + MVR_PTS + BLUEBOOK_1000 + TRAVTIME
y = policy_train['CLAIM_FLAG']
designX = policy_train[['MVR_PTS','BLUEBOOK_1000','TRAVTIME']]
designX = stats.add_constant(designX, prepend=True)
LLK0, DF0, fullParams0, thisFit = build_mnlogit (designX, y)

# Apply the Final Model to the Testing partition
X = policy_test[['MVR_PTS','BLUEBOOK_1000','TRAVTIME']]
X = stats.add_constant(X, prepend=True)

yPredProb = thisFit.predict(X)
y = policy_test['CLAIM_FLAG']

# Calculate the Area Under Curve value for the Testing partition
testAUC = metrics.roc_auc_score(y, yPredProb[1])