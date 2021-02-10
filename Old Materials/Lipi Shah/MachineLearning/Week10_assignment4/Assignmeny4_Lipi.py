# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:40:53 2020

@author: Lipi
"""

## Note:- I have taken reference from the professor sample code for some part of assignment.
########################### Import Statements #################################
#########################################Question 1 ########################################
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as stats
import sympy
import math


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


def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)


purchaseLikelihood = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week10_assignment4\\Purchase_Likelihood.csv', delimiter=',',
                                      usecols=['group_size', 'homeowner', 'married_couple', 'insurance'])

y = purchaseLikelihood['insurance'].astype('category')


x_gs = pd.get_dummies(purchaseLikelihood[['group_size']].astype('category'))
x_mc = pd.get_dummies(purchaseLikelihood[['married_couple']].astype('category'))
x_ho = pd.get_dummies(purchaseLikelihood[['homeowner']].astype('category'))


# Intercept only model effect 0
design_x = pd.DataFrame(y.where(y.isnull(), 1))
llk0, df0, full_params0 = build_mnlogit(design_x, y, debug='Y')

# Intercept + group_size effect 1
design_x = stats.add_constant(x_gs, prepend=True)
llk1_gs, df1_gs, full_params1_gs = build_mnlogit (design_x, y, debug = 'Y')
testDev_gs = 2 * (llk1_gs - llk0)
testDF_gs = df1_gs - df0
testPValue_gs = scipy.stats.chi2.sf(testDev_gs, testDF_gs)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs)
print('  Degrees of Freedom = ', testDF_gs)
print('        Significance = ', testPValue_gs)


# Intercept + group_size + homeowner effect 2
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


# Intercept + group_size + homeowner + married_couple effect 3
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




#########
# Intercept + group_size + homeowner + married_couple + group_size * homeowner effect 4
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



############# effect 5

# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)

# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the group_size * married_couple interaction effect
x_gsmc = create_interaction(x_gs, x_mc)
design_x = design_x.join(x_gsmc)
design_x = stats.add_constant(design_x, prepend=True)

llk5_gs_ho_mc_gsho_gsmc, df5_gs_ho_mc_gsho_gsmc, full_params5_gs_ho_mc_gsho_gsmc = build_mnlogit(design_x, y, debug='Y')
testDev_gs_ho_mc_gsho_gsmc = 2 * (llk5_gs_ho_mc_gsho_gsmc - llk4_gs_ho_mc_gsho)
testDF_gs_ho_mc_gsho_gsmc = df5_gs_ho_mc_gsho_gsmc - df4_gs_ho_mc_gsho
testPValue_gs_ho_mc_gsho_gsmc = scipy.stats.chi2.sf(testDev_gs_ho_mc_gsho_gsmc, testDF_gs_ho_mc_gsho_gsmc)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho_gsmc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho_gsmc)
print('        Significance = ', testPValue_gs_ho_mc_gsho_gsmc)


#########  effect 6
# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple 
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)

# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the group_size * married_couple interaction effect
x_gsmc = create_interaction(x_gs, x_mc)
design_x = design_x.join(x_gsmc)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the homeowner * married_couple  interaction effect
x_homc = create_interaction(x_ho, x_mc)
design_x = design_x.join(x_homc)
design_x = stats.add_constant(design_x, prepend=True)


llk6_gs_ho_mc_gsho_gsmc_homc, df6_gs_ho_mc_gsho_gsmc_homc, full_params6_gs_ho_mc_gsho_gsmc_homc = build_mnlogit(design_x, y, debug='Y')
testDev_gs_ho_mc_gsho_gsmc_homc = 2 * (llk6_gs_ho_mc_gsho_gsmc_homc - llk5_gs_ho_mc_gsho_gsmc)
testDF_gs_ho_mc_gsho_gsmc_homc = df6_gs_ho_mc_gsho_gsmc_homc - df5_gs_ho_mc_gsho_gsmc
testPValue_gs_ho_mc_gsho_gsmc_homc = scipy.stats.chi2.sf(testDev_gs_ho_mc_gsho_gsmc_homc, testDF_gs_ho_mc_gsho_gsmc_homc)

print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho_gsmc_homc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho_gsmc_homc)
print('        Significance = ', testPValue_gs_ho_mc_gsho_gsmc_homc)


print("#########################################Question 1a ########################################")

print("The aliased parameters are those for which corresponding values are zero from the below. I have mentioned in the document")
print(full_params6_gs_ho_mc_gsho_gsmc_homc)

print("#########################################Question 1b ########################################")

print("Degree of Freedom =",testDF_gs_ho_mc_gsho_gsmc_homc)


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
    chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(purchaseLikelihood[pred], purchaseLikelihood['insurance'], debug = 'Y')
    test_result.loc[pred] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]
    
for pred in intPred:
    devianceStat, devianceDf, devianceSig, mcFaddenRSq = DevianceTest(purchaseLikelihood[pred], purchaseLikelihood['insurance'], debug = 'Y')
    test_result.loc[pred] = ['Deviance', devianceStat, devianceDf, devianceSig, 'McFadden''s R^2', mcFaddenRSq]

print(test_result[test_result['Test'] == 'Deviance'])



print("#########################################Question 1c ########################################")

# Intercept 
print("For model:Intercept  ")
#llk0, df0
print('Free Parameter = ', df0)
print('  Log-Likelihood = ', llk0)


# Intercept + group_size
print("For model:Intercept + group_size ")
#llk1_gs, df1_gs
print('Free Parameter = ', df1_gs)
print('  Log-Likelihood = ', llk1_gs)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs)
print('  Degrees of Freedom = ', testDF_gs)
print('        Significance = ', testPValue_gs)


# Intercept + group_size + homeowner
print("For model:Intercept + group_size + homeowner")
#llk2_gs_ho, df2_gs_ho
print('Free Parameter = ', df2_gs_ho)
print('  Log-Likelihood = ', llk2_gs_ho)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho)
print('  Degrees of Freedom = ', testDF_gs_ho)
print('        Significance = ', testPValue_gs_ho)


# Intercept + group_size + homeowner + married_couple
print("For model:Intercept + group_size + homeowner + married_couple")
#llk3_gs_ho_mc, df3_gs_ho_mc
print('Free Parameter = ', df3_gs_ho_mc)
print('  Log-Likelihood = ', llk3_gs_ho_mc)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc)
print('        Significance = ', testPValue_gs_ho_mc)




# Intercept + group_size + homeowner + married_couple + group_size * homeowner
print("For model:Intercept + group_size + homeowner + married_couple + group_size * homeowner")
#llk4_gs_ho_mc_gsho, df4_gs_ho_mc_gsho
print('Free Parameter = ', df4_gs_ho_mc_gsho)
print('  Log-Likelihood = ', llk4_gs_ho_mc_gsho)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho)
print('        Significance = ', testPValue_gs_ho_mc_gsho)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple
print("For model:Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple")
#llk5_gs_ho_mc_gsho_gsmc, df5_gs_ho_mc_gsho_gsmc
print('Free Parameter = ', df5_gs_ho_mc_gsho_gsmc)
print('  Log-Likelihood = ', llk5_gs_ho_mc_gsho_gsmc)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho_gsmc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho_gsmc)
print('        Significance = ', testPValue_gs_ho_mc_gsho_gsmc)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple
print("For model:Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple")
#llk6_gs_ho_mc_gsho_gsmc_homc, df6_gs_ho_mc_gsho_gsmc_homc
print('Free Parameter = ', df6_gs_ho_mc_gsho_gsmc_homc)
print('  Log-Likelihood = ', llk6_gs_ho_mc_gsho_gsmc_homc)
print('Deviance Chi-Square Test')
print('Chi-Square Statistic = ', testDev_gs_ho_mc_gsho_gsmc_homc)
print('  Degrees of Freedom = ', testDF_gs_ho_mc_gsho_gsmc_homc)
print('        Significance = ', testPValue_gs_ho_mc_gsho_gsmc_homc)


print("#########################################Question 1d ########################################")
FI1 = -(math.log10(testPValue_gs))
FI2 = 0 #-(math.log10(testPValue_gs_ho))
FI3 = -(math.log10(testPValue_gs_ho_mc))
FI4 = -(math.log10(testPValue_gs_ho_mc_gsho))
FI5 = -(math.log10(testPValue_gs_ho_mc_gsho_gsmc))
FI6 =-(math.log10(testPValue_gs_ho_mc_gsho_gsmc))

print("Feature Importance Index for model:(Intercept + group_size) =", FI1)
print("Feature Importance Index for model:(Intercept + group_size + homeowner) = Undefined", FI2)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple) =", FI3)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner) =",FI4)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple) =", FI5)
print("Feature Importance Index for model:(Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple) =", FI6)




#print("#########################################Question 2A ########################################")

design_x_old = design_x
logit = stats.MNLogit(y, design_x)


design_x = design_x_old

design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)

# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the group_size * married_couple interaction effect
x_gsmc = create_interaction(x_gs, x_mc)
design_x = design_x.join(x_gsmc)
design_x = stats.add_constant(design_x, prepend=True)

# Create the columns for the homeowner * married_couple  interaction effect
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

# Create the columns for the group_size * married_couple interaction effect
x_gsmc = create_interaction(x_gs, x_mc)
x_design = x_design.join(x_gsmc)
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



print("#########################################Question 2A ########################################")

Final = []
GS = [1,2,3,4]
H = [0,1]
M = [0,1]

for i in GS:
    for j in H:
        for k in M:
            Final.append([i,j,k])

df = pd.DataFrame(Final, columns=['group_size','homeowner','married_couple'])

df_groupsize = pd.get_dummies(df[['group_size']].astype('category'))
FinalX = df_groupsize

df_homeowner = pd.get_dummies(df[['homeowner']].astype('category'))
FinalX = FinalX.join(df_homeowner)

df_marriedcouple = pd.get_dummies(df[['married_couple']].astype('category'))
FinalX = FinalX.join(df_marriedcouple)

df_groupsize_h = create_interaction(df_groupsize, df_homeowner)
df_groupsize_h = pd.get_dummies(df_groupsize_h)
FinalX = FinalX.join(df_groupsize_h)

df_groupsize_m = create_interaction(df_groupsize, df_marriedcouple)
df_groupsize_m = pd.get_dummies(df_groupsize_m)
FinalX = FinalX.join(df_groupsize_m)

df_homeowner_m = create_interaction(df_homeowner, df_marriedcouple)
df_homeowner_m = pd.get_dummies(df_homeowner_m)
FinalX = FinalX.join(df_homeowner_m)


FinalX = stats.add_constant(FinalX, prepend=True)

logit = stats.MNLogit(y, design_x_old)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)           
PP = thisFit.predict(FinalX)
print(PP)

import pandas as pd
df1=pd.DataFrame(PP)
df2=pd.DataFrame(Final)
result= pd.merge(df2,df1, left_index=True,right_index=True)

result1=result.rename(columns={'0_x':'Group_size','1_x':'Homeowner','2_x':'married_couple','0_y':'Prob(Insurance=0)','1_y':'Prob(Insurance=1)','2_y':'Prob(Insurance=2)'})
result1
pd_result = pd.DataFrame(result1)


print("############## Answer 2 B #####################")
max = PP[1]/PP[0]
print(max.max(axis = 0))
#
print("############## Answer 2 C #####################")
#Taking insurance =0 as reference target category
# 	Loge((Prob(insurance =2)/Prob(insurance =0) | group_size = 3) ) – 
#loge((Prob(insurance =2)/Prob(insurance =0) | group_size = 1))
# = Parameter of (group_size = 3 | A=2) – Parameter of (group_size = 1 | A=2)
#               = (0.503430 - 0.546053)+(-0.524617 + 0.880455) + (0.337205 - 0.902886)
#               = -0.25246599999999997
# Taking exponent of the previous value: exp(-0.25246599999999997) = 0.776882626399578
print(math.exp(-0.25246599999999997))
print("############## Answer 2 D #####################")
#(Prob(insurance =0)/Prob(insurance =1) | homeowner = 1) / ((Prob(insurance =0)/Prob(insurance =1) | homeowner = 0)
#
#= Log (Prob(insurance =0)/Prob(insurance =1) | homeowner = 1) - log((Prob(insurance =0)/Prob(insurance =1) | homeowner = 0)
#=(0-0.776052)+(0 + 1.395311) + (0 + 1.086733) + ( 0 + 0.635960) + (0 - 0.115368)
#=1 /Exp (2.226584)
print(math.exp(-2.226584))
    
print("################ Answer 3 A #########################")
import numpy as np
import pandas as pd
import scipy
from sklearn import naive_bayes 

purchaseLikelihood = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\\Week10_assignment4\\Purchase_Likelihood.csv')

countTarget = pd.DataFrame(columns = ['Count', 'Class_probability'])
countTarget['Count'] = purchaseLikelihood['insurance'].value_counts()
countTarget['Class_probability'] = purchaseLikelihood['insurance'].value_counts(normalize=True)
print(countTarget)



print("################ Answer 3 B C D #########################")

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

     
      
print("################ Answer 3 E #########################")
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


print("################ Answer 3 F #########################")
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
y_test_pred_prob = pd.DataFrame(_objNB.predict_proba(x_test), columns = ['Prob(insurance = 0)', 'Prob(insurance = 1)','Prob(insurance = 2)'])
y_test_score = pd.concat([x_test, y_test_pred_prob], axis = 1)
                                                                                      
print(y_test_score)
answer_dataframe = pd.DataFrame(y_test_score) 


print("################ Answer 3 G #########################")
max1 = y_test_score["Prob(insurance = 1)"] / y_test_score["Prob(insurance = 0)"]
y_test_score["a1/a0"] = max1
print(y_test_score)
print(y_test_score.loc[[3]])
print(max1.max(axis = 0))