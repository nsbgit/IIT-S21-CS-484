import pandas as pd
import statsmodels.api as stats
import sympy
import scipy
import numpy

def create_interaction(in_df1, in_df2):
    name1 = in_df1.columns
    name2 = in_df2.columns
    out_df = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            out_df[outName] = in_df1[col1] * in_df2[col2]
    return (out_df)

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

    # Recreate the estimates of the full parameters
    workParams = pd.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

purchase_ll = pd.read_csv('Purchase_Likelihood.csv')
purchase_ll = purchase_ll.dropna()

# Specify Origin as a categorical variable
y = purchase_ll['insurance'].astype('category')

# Specify GROUP_SIZE, HOMEOWNER and MARRIED_COUPLE as categorical variables
xgs = pd.get_dummies(purchase_ll[['group_size']].astype('category'))
xho = pd.get_dummies(purchase_ll[['homeowner']].astype('category'))
xmc = pd.get_dummies(purchase_ll[['married_couple']].astype('category'))

# Intercept only model
designX = pd.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0 = build_mnlogit (designX, y, debug = 'Y')
print("*"*50)

# Intercept + GROUP_SIZE
designX = stats.add_constant(xgs, prepend=True)
LLK_1G, DF_1G, fullParams_1G = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G - LLK0)
testDF = DF_1G - DF0
testPValue_GS = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print("Number of Free Parameters =", DF_1G)
print("Model Log-Likelihood Value =", LLK_1G)
print('Deviance test Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue_GS)
print("*"*50)

# Intercept + GROUP_SIZE + HOMEOWNER
designX = xgs
designX = designX.join(xho)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H, DF_1G_1H, fullParams_1G_1H = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H - LLK_1G)
testDF = DF_1G_1H - DF_1G
testPValue_GS_HO = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print("Number of Free Parameters =", DF_1G_1H)
print("Model Log-Likelihood Value =", LLK_1G_1H)
print('Deviance test Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue_GS_HO)
print("*"*50)

# Intercept + GROUP_SIZE + HOMEOWNER + MARRIED_COUPLE
designX = xgs
designX = designX.join(xho)
designX = designX.join(xmc)
designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M, DF_1G_1H_1M, fullParams_1G_1H_1M = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M - LLK_1G_1H)
testDF = DF_1G_1H_1M - DF_1G_1H
testPValue_GS_HO_MC = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print("Number of Free Parameters =", DF_1G_1H_1M)
print("Model Log-Likelihood Value =", LLK_1G_1H_1M)
print('Deviance test Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue_GS_HO_MC)
print("*"*50)

# Intercept + GROUP_SIZE + HOMEOWNER + MARRIED_COUPLE + GROUP_SIZE * HOMEOWNER
designX = xgs
designX = designX.join(xho)
designX = designX.join(xmc)

# Create the columns for the GROUP_SIZE * HOMEOWNER interaction effect
xgsho = create_interaction(xgs, xho)
designX = designX.join(xgsho)

designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M_1GSHO, DF_1G_1H_1M_1GSHO, fullParams_1G_1H_1M_1GSHO = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M_1GSHO - LLK_1G_1H_1M)
testDF = DF_1G_1H_1M_1GSHO - DF_1G_1H_1M
testPValue_GS_HO_MC_GSHO = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print("Number of Free Parameters =", DF_1G_1H_1M_1GSHO)
print("Model Log-Likelihood Value =", LLK_1G_1H_1M_1GSHO)
print('Deviance test Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue_GS_HO_MC_GSHO)
print("*"*50)

# Intercept + GROUP_SIZE + HOMEOWNER + MARRIED_COUPLE + GROUP_SIZE * HOMEOWNER + GROUP_SIZE * MARRIED_COUPLE
designX = xgs
designX = designX.join(xho)
designX = designX.join(xmc)
designX = designX.join(xgsho)

# Create the columns for the GROUP_SIZE * MARRIED_COUPLE interaction effect
xgsmc = create_interaction(xgs, xmc)
designX = designX.join(xgsmc)

designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M_1GSHO_1GSMC, DF_1G_1H_1M_1GSHO_1GSMC, fullParams_1G_1H_1M_1GSHO_1GSMC = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M_1GSHO_1GSMC - LLK_1G_1H_1M_1GSHO)
testDF = DF_1G_1H_1M_1GSHO_1GSMC - DF_1G_1H_1M_1GSHO
testPValue_GS_HO_MC_GSHO_GSMC = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print("Number of Free Parameters =", DF_1G_1H_1M_1GSHO_1GSMC)
print("Model Log-Likelihood Value =", LLK_1G_1H_1M_1GSHO_1GSMC)
print('Deviance test Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue_GS_HO_MC_GSHO_GSMC)
print("Deviance (Statistic, DF, Significance)", testDev, testDF, testPValue_GS_HO_MC_GSHO_GSMC)
print("*"*50)

# Intercept + GROUP_SIZE + HOMEOWNER + MARRIED_COUPLE + GROUP_SIZE * HOMEOWNER + GROUP_SIZE * MARRIED_COUPLE + HOMEOWNER * MARRIED_COUPLE
designX = xgs
designX = designX.join(xho)
designX = designX.join(xmc)
designX = designX.join(xgsho)
designX = designX.join(xgsmc)

# Create the columns for the HOMEOWNER * MARRIED_COUPLE interaction effect
xhomc = create_interaction(xho, xmc)
designX = designX.join(xhomc)

designX = stats.add_constant(designX, prepend=True)
LLK_1G_1H_1M_1GSHO_1GSMC_1HOMC, DF_1G_1H_1M_1GSHO_1GSMC_1HOMC, fullParams_1G_1H_1M_1GSHO_1GSMC_1HOMC = build_mnlogit (designX, y, debug = 'N')
testDev = 2 * (LLK_1G_1H_1M_1GSHO_1GSMC_1HOMC - LLK_1G_1H_1M_1GSHO_1GSMC)
testDF = DF_1G_1H_1M_1GSHO_1GSMC_1HOMC - DF_1G_1H_1M_1GSHO_1GSMC
testPValue_GS_HO_MC_GSHO_GSMC_HOMC = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print("Number of Free Parameters =", DF_1G_1H_1M_1GSHO_1GSMC_1HOMC)
print("Model Log-Likelihood Value =", LLK_1G_1H_1M_1GSHO_1GSMC_1HOMC)
print('Deviance test Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue_GS_HO_MC_GSHO_GSMC_HOMC)
print("*"*50)

print("Q1.a)(5 points) List the aliased columns that you found in your model matrix")
print(fullParams_1G_1H_1M_1GSHO_1GSMC_1HOMC.loc[fullParams_1G_1H_1M_1GSHO_1GSMC_1HOMC['0_y'] == 0.0].index)
print("*"*50)

print("Q1.b)(5 points) How many degrees of freedom does your model have?")
print(f'Degree of Freedom = {testDF}')
print("*"*50)

print("Q1.d)(5 points) Calculate the Feature Importance Index as the negative base-10 logarithm of the significance value.  List your indices by the model effects.")
print(f'Feature Importance Index for (Intercept + group_size) = {-(numpy.log10(testPValue_GS))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner) = {-(numpy.log10(testPValue_GS_HO))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple) = {-(numpy.log10(testPValue_GS_HO_MC))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner) = {-(numpy.log10(testPValue_GS_HO_MC_GSHO))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple) = {-(numpy.log10(testPValue_GS_HO_MC_GSHO_GSMC))}')
print(f'Feature Importance Index for (Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple) = {-(numpy.log10(testPValue_GS_HO_MC_GSHO_GSMC_HOMC))}')