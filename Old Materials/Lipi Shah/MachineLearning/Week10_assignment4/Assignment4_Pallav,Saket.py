## Note:- i have taken refence from professor's lecture slide for some parts of the code.

#########################################Question 1 ########################################
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as stats
import sympy
import math

def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)


def build_mnlogit(full_x, y, debug='N'):
    # Number of all parameters
    nFullParam = full_x.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(full_x.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    x = full_x.iloc[:, list(inds)]

    # The number of free parameters
    thisDf = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, x)
    this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
    this_parameter = this_fit.params
    this_llk = logit.loglike(this_parameter.values)

    if (debug == 'Y'):
        print(this_fit.summary())
        print("Model Parameter Estimates:\n", this_parameter)
        print("Model Log-Likelihood Value =", this_llk)
        print("Number of Free Parameters =", thisDf)

    # Recreat the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape=(nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys=full_x.columns)
    fullParams = pd.merge(workParams, this_parameter, how="left", left_index=True, right_index=True)
    fullParams = fullParams.drop(columns='0_x').fillna(0.0)

    # Return model statistics
    return (this_llk, thisDf, fullParams)



purchaseLikelihood = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week10_assignment4\\Purchase_Likelihood.csv', delimiter=',',
                                      usecols=['group_size', 'homeowner', 'married_couple', 'A'])


y = purchaseLikelihood['A'].astype('category')

x_gs = pd.get_dummies(purchaseLikelihood[['group_size']].astype('category'))
x_ho = pd.get_dummies(purchaseLikelihood[['homeowner']].astype('category'))
x_mc = pd.get_dummies(purchaseLikelihood[['married_couple']].astype('category'))

# Intercept only model
design_x = pd.DataFrame(y.where(y.isnull(), 1))
llk0, df0, full_params0 = build_mnlogit(design_x, y, debug='Y')


# Intercept + group_size
design_x = stats.add_constant(x_gs, prepend=True)
llk1_gs, df1_gs, full_params1_gs = build_mnlogit (design_x, y, debug = 'Y')
testDev_gs = 2 * (llk1_gs - llk0)
testDF_gs = df1_gs - df0
testPValue_gs = scipy.stats.chi2.sf(testDev_gs, testDF_gs)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs)
print('  Degrees of Freedom = ', testDF_gs)
print('        Significance = ', testPValue_gs)


# Intercept + group_size + homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = stats.add_constant(design_x, prepend=True)
llk2_gs_ho, df2_gs_ho, full_params2_gs_ho = build_mnlogit (design_x, y, debug = 'Y')
testDev_gs_ho = 2 * (llk2_gs_ho - llk1_gs)
testDF_gs_ho = df2_gs_ho - df1_gs
testPValue_gs_ho = scipy.stats.chi2.sf(testDev_gs_ho, testDF_gs_ho)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho)
print('  Degrees of Freedom = ', testDF_gs_ho)
print('        Significance = ', testPValue_gs_ho)


# Intercept + group_size + homeowner + married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
design_x = stats.add_constant(design_x, prepend=True)
llk3_gs_ho_mc, df3_gs_ho_mc, full_params3_gs_ho_mc = build_mnlogit (design_x, y, debug = 'Y')
testDev_gs_ho_mc = 2 * (llk3_gs_ho_mc - llk2_gs_ho)
testDF_gs_ho_mc = df3_gs_ho_mc - df2_gs_ho
testPValue_gs_ho_mc = scipy.stats.chi2.sf(testDev_gs_ho_mc, testDF_gs_ho_mc)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc)
print('        Significance = ', testPValue_gs_ho_mc)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)

# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
llk4_gs_ho_mc_gsho, df4_gs_ho_mc_gsho, full_params4_gs_ho_mc_gsho = build_mnlogit(design_x, y, debug='Y')
testDev_gs_ho_mc_gsho = 2 * (llk4_gs_ho_mc_gsho - llk3_gs_ho_mc)
testDF_gs_ho_mc_gsho = df4_gs_ho_mc_gsho - df3_gs_ho_mc
testPValue_gs_ho_mc_gsho = scipy.stats.chi2.sf(testDev_gs_ho_mc_gsho, testDF_gs_ho_mc_gsho)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho)
print('        Significance = ', testPValue_gs_ho_mc_gsho)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)

# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
design_x = design_x.join(x_homc)
design_x = stats.add_constant(design_x, prepend=True)
llk5_gs_ho_mc_gsho_homc, df5_gs_ho_mc_gsho_homc, full_params5_gs_ho_mc_gsho_homc = build_mnlogit(design_x, y, debug='Y')
testDev_gs_ho_mc_gsho_homc = 2 * (llk5_gs_ho_mc_gsho_homc - llk4_gs_ho_mc_gsho)
testDF_gs_ho_mc_gsho_homc = df5_gs_ho_mc_gsho_homc - df4_gs_ho_mc_gsho
testPValue_gs_ho_mc_gsho_homc = scipy.stats.chi2.sf(testDev_gs_ho_mc_gsho_homc, testDF_gs_ho_mc_gsho_homc)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho_homc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho_homc)
print('        Significance = ', testPValue_gs_ho_mc_gsho_homc)


print("#########################################Question 1a ########################################")

print("The aliased parameters are those for which corresponding values are zero from the below. I have mentioned in the document")
print(full_params5_gs_ho_mc_gsho_homc)


print("#########################################Question 1b ########################################")

print("Degree of Freedom =",df5_gs_ho_mc_gsho_homc)
# I think it is : testDF_gs_ho_mc_gsho_homc

def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))

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
    cramerV = np.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)
    

def DevianceTest (
    xInt,           # input interval feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    y = yCat.astype('category')

    # Model 0 is yCat = Intercept
    X = np.where(yCat.notnull(), 1, 0)
    objLogit = stats.MNLogit(y, X)
    thisFit = objLogit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    LLK0 = objLogit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print(thisFit.summary())
        print("Model Log-Likelihood Value =", LLK0)
        print('\n')

    # Model 1 is yCat = Intercept + xInt
    X = stats.add_constant(xInt, prepend = True)
    objLogit = stats.MNLogit(y, X)
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

predictors = ['group_size', 'homeowner', 'married_couple']
intPred = predictors

test_result = pd.DataFrame(index = predictors + intPred, columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in predictors:
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(purchaseLikelihood[pred], purchaseLikelihood['A'], debug = 'Y')
    test_result.loc[pred] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]
    
for pred in intPred:
    devianceStat, devianceDf, devianceSig, mcFaddenRSq = DevianceTest(purchaseLikelihood[pred], purchaseLikelihood['A'], debug = 'Y')
    test_result.loc[pred] = ['Deviance', devianceStat, devianceDf, devianceSig, 'McFadden''s R^2', mcFaddenRSq]

print(test_result[test_result['Test'] == 'Deviance'])

print("#########################################Question 1c ########################################")

# Intercept + group_size
print("For model:Intercept + group_size ")
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs)
print('  Degrees of Freedom = ', testDF_gs)
print('        Significance = ', testPValue_gs)

# Intercept + group_size + homeowner
print("For model:Intercept + group_size + homeowner")
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho)
print('  Degrees of Freedom = ', testDF_gs_ho)
print('        Significance = ', testPValue_gs_ho)

# Intercept + group_size + homeowner + married_couple
print("For model:Intercept + group_size + homeowner + married_couple")
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc)
print('        Significance = ', testPValue_gs_ho_mc)
      
# Intercept + group_size + homeowner + married_couple + group_size * homeowner
print("For model:Intercept + group_size + homeowner + married_couple + group_size * homeowner")
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho)
print('        Significance = ', testPValue_gs_ho_mc_gsho)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple
print("For model:Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple")
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho_homc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho_homc)
print('        Significance = ', testPValue_gs_ho_mc_gsho_homc)

print("#########################################Question 1d ########################################")
FI1 = -(math.log10(testPValue_gs))
FI2 = 0 #-(math.log10(testPValue_gs_ho))
FI3 = -(math.log10(testPValue_gs_ho_mc))
FI4 = -(math.log10(testPValue_gs_ho_mc_gsho))
FI5 = -(math.log10(testPValue_gs_ho_mc_gsho_homc))
print("Feature Importance Index for model:(Intercept + group_size) =", FI1)
#print('Feature Importance Index for model:(Intercept + group_size + homeowner) = {-(math.log10(testPValue_gs_ho))}')
print("Feature Importance Index for model:(Intercept + group_size + homeowner) =", FI2)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple) =", FI3)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner) =",FI4)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner + homeowner * married_couple) =", FI5)


print("#########################################Question 1e ########################################")
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
design_x = design_x.join(x_homc)
design_x = stats.add_constant(design_x, prepend=True)

logit = stats.MNLogit(y, design_x)
this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

gs_d = [1,2,3,4]
ho_d = [0,1]
mc_d = [0,1]
A_d = [0,1,2]

x_data = []

for i in gs_d:
    for j in ho_d:
        for k in mc_d:
            data = [i,j,k]
            x_data = x_data + [data]

x_input = pd.DataFrame(x_data, columns=['group_size','homeowner','married_couple'])
x_gs = pd.get_dummies(x_input[['group_size']].astype('category'))
x_ho = pd.get_dummies(x_input[['homeowner']].astype('category'))
x_mc = pd.get_dummies(x_input[['married_couple']].astype('category'))
x_design = x_gs
x_design = x_design.join(x_ho)
x_design = x_design.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
x_design = x_design.join(x_gsho)
x_design = stats.add_constant(x_design, prepend=True)
# Create the columns for the homeowner * married_couple interaction effect
x_homc = create_interaction(x_ho, x_mc)
x_design = x_design.join(x_homc)
x_design = stats.add_constant(x_design, prepend=True)
A_pred = this_fit.predict(exog = x_design)
A_pred = pd.DataFrame(A_pred, columns = ['PA0', 'PA1','PA2'])

A_output = pd.concat([x_input, A_pred],axis=1)
print(A_output)

A_output['odd value(PA1/PA0)'] = A_output['PA1'] / A_output['PA0']
print(A_output[['group_size','homeowner','married_couple','odd value(PA1/PA0)']])
print(A_output.loc[A_output['odd value(PA1/PA0)'].idxmax()])

print("Other parts of question 1 are mention in the document")

############################################################################################
#########################################Question 2 ########################################
############################################################################################
import numpy as np
import pandas as pd
import scipy
from sklearn import naive_bayes 

purchaseLikelihood = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week10_assignment4\\Purchase_Likelihood.csv')

print("#########################################Question 2a ########################################")
countTarget = pd.DataFrame(columns = ['Count', 'Class_probability'])
countTarget['Count'] = purchaseLikelihood['insurance'].value_counts()
countTarget['Class_probability'] = purchaseLikelihood['insurance'].value_counts(normalize=True)
print(countTarget)


print("#########################################Question 2 b c d ########################################")


def crossTabulation(row_var, col_var, show = 'ROW'):

    countTable = pd.crosstab(index = row_var, columns = col_var)
    print("Frequency: \n", countTable)

    if (show == 'ROW' or show == 'BOTH'):
        RowF = countTable.div(countTable.sum(1), axis='index')
        print("Row Table: \n", RowF)

    if (show == 'COLUMN' or show == 'BOTH'):
        ColF = countTable.div(countTable.sum(0), axis='columns')
        print("Column Frac Table: \n", ColF)

    return

for col in purchaseLikelihood.columns[:-1]:
    crossTabulation(purchaseLikelihood['insurance'],purchaseLikelihood[col],'ROW')

def ChiSquareTest (
    xCat,           # input categorical feature
    yCat,           # input categorical target variable
    debug = 'N'     # debugging flag (Y/N) 
    ):

    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))

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
    cramerV = np.sqrt(cramerV)

    return(chiSqStat, chiSqDf, chiSqSig, cramerV)


catPred = ['group_size', 'homeowner', 'married_couple']
intPred = []
testResult = pd.DataFrame(index = catPred + intPred, columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for col in purchaseLikelihood.columns[:-1]:
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(purchaseLikelihood[col], purchaseLikelihood['insurance'], debug = 'Y')
    testResult.loc[col] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]

print("#########################################Question 2 e ########################################")
print(testResult.sort_values('Measure'))
print('\n homeowner has the largest association with the target variable')

xTrain = purchaseLikelihood[catPred].astype('category')
yTrain = purchaseLikelihood['insurance'].astype('category')

_objNB = naive_bayes.MultinomialNB(alpha = 1.0e-10)
thisFit = _objNB.fit(xTrain, yTrain)

print('Probability of each class',np.exp(thisFit.class_log_prior_))
print('Empirical probability of features given a class, P(x_i|y)',np.exp(thisFit.feature_log_prob_))
print('Number of samples encountered for each class during fitting',thisFit.class_count_)
print('Number of samples encountered for each (class, feature) during fitting',thisFit.feature_count_)

print("#########################################Question 2 g ########################################")
# Create the all possible combinations of the features' values
gs_d = [1,2,3,4]
ho_d = [0,1]
mc_d = [0,1]
A_d = [0,1,2]

final_data = []

for i in gs_d:
    for j in ho_d:
        for k in mc_d:
            data = [i,j,k]
            final_data = final_data + [data]

x_test = pd.DataFrame(final_data, columns=['group_size','homeowner','married_couple'])
x_test = x_test[catPred].astype('category')
y_test_pred_prob = pd.DataFrame(_objNB.predict_proba(x_test), columns = ['PA0', 'PA1','PA2'])
y_test_score = pd.concat([x_test, y_test_pred_prob], axis = 1)
                                                                                      
print(y_test_score)

print("Rest parts of question 2 is mention in the document")