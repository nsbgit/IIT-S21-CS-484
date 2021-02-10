import numpy
import pandas
import scipy
import sympy 
import statsmodels.api as stats
import sklearn.ensemble as ensemble


# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction(in_df1, in_df2):
    name1 = in_df1.columns
    name2 = in_df2.columns
    out_df = pandas.DataFrame()
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

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)

purchase_likelihood = pandas.read_csv('D:\\IIT Edu\\Sem1\\MachineLearning\\Week10_assignment4\\Purchase_Likelihood.csv', delimiter=',',
                                      usecols=['group_size', 'homeowner', 'married_couple', 'A'])
purchase_likelihood = purchase_likelihood.dropna()

no_objs = purchase_likelihood.shape[0]

y = purchase_likelihood['A'].astype('category')

x_gs = pandas.get_dummies(purchase_likelihood[['group_size']].astype('category'))
x_ho = pandas.get_dummies(purchase_likelihood[['homeowner']].astype('category'))
x_mc = pandas.get_dummies(purchase_likelihood[['married_couple']].astype('category'))

# Intercept only model
design_x = pandas.DataFrame(y.where(y.isnull(), 1))
llk0, df0, full_params0 = build_mnlogit(design_x, y, debug='Y')

# Intercept + group_size
design_x = stats.add_constant(x_gs, prepend=True)
llk1_gs, df1_gs, full_params1_gs = build_mnlogit (design_x, y, debug = 'Y')
test_dev = 2 * (llk1_gs - llk0)
test_df = df1_gs - df0
test_p_value = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value)

# Intercept + group_size + homeowner

design_x = x_gs
design_x = design_x.join(x_ho)
design_x = stats.add_constant(design_x, prepend=True)
llk2_gs_ho, df2_gs_ho, full_params2_gs_ho = build_mnlogit (design_x, y, debug = 'Y')
test_dev = 2 * (llk2_gs_ho - llk1_gs)
test_df = df2_gs_ho - df1_gs
test_p_value = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value)

# Intercept + group_size + homeowner + married_couple
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
design_x = stats.add_constant(design_x, prepend=True)
llk3_gs_ho_mc, df3_gs_ho_mc, full_params3_gs_ho_mc = build_mnlogit (design_x, y, debug = 'Y')
test_dev = 2 * (llk3_gs_ho_mc - llk2_gs_ho)
test_df = df3_gs_ho_mc - df2_gs_ho
test_p_value = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner
design_x = x_gs
design_x = design_x.join(x_ho)
design_x = design_x.join(x_mc)
# Create the columns for the group_size * homeowner interaction effect
x_gsho = create_interaction(x_gs, x_ho)
design_x = design_x.join(x_gsho)
design_x = stats.add_constant(design_x, prepend=True)
llk4_gs_ho_mc_gsho, df4_gs_ho_mc_gsho, full_params4_gs_ho_mc_gsho = build_mnlogit(design_x, y, debug='Y')
test_dev = 2 * (llk4_gs_ho_mc_gsho - llk3_gs_ho_mc)
test_df = df4_gs_ho_mc_gsho - df3_gs_ho_mc
test_p_value = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value)

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
test_dev = 2 * (llk5_gs_ho_mc_gsho_homc - llk4_gs_ho_mc_gsho)
test_df = df5_gs_ho_mc_gsho_homc - df4_gs_ho_mc_gsho
test_p_value = scipy.stats.chi2.sf(test_dev, test_df)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', test_dev)
print('  Degrees of Freedom = ', test_df)
print('        Significance = ', test_p_value)

def chi_square_test (x_cat, y_cat, debug = 'N'):

    obs_count = pandas.crosstab(index = x_cat, columns = y_cat, margins = False, dropna = True)
    c_total = obs_count.sum(axis = 1)
    r_total = obs_count.sum(axis = 0)
    n_total = numpy.sum(r_total)
    exp_count = numpy.outer(c_total, (r_total / n_total))

    if (debug == 'Y'):
        print('Observed Count:\n', obs_count)
        print('Column Total:\n', c_total)
        print('Row Total:\n', r_total)
        print('Overall Total:\n', n_total)
        print('Expected Count:\n', exp_count)
        print('\n')
       
    chi_sq_stat = ((obs_count - exp_count)**2 / exp_count).to_numpy().sum()
    chi_sq_df = (obs_count.shape[0] - 1.0) * (obs_count.shape[1] - 1.0)
    chi_sq_sig = scipy.stats.chi2.sf(chi_sq_stat, chi_sq_df)

    cramer_v = chi_sq_stat / n_total
    if (c_total.size > r_total.size):
        cramer_v = cramer_v / (r_total.size - 1.0)
    else:
        cramer_v = cramer_v / (c_total.size - 1.0)
    cramer_v = numpy.sqrt(cramer_v)

    return(chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v)
    
def deviance_test (x_int, y_cat, debug = 'N' ):

    y = y_cat.astype('category')

    # Model 0 is yCat = Intercept
    x = numpy.where(y_cat.notnull(), 1, 0)
    obj_logit = smodel.MNLogit(y, x)
    this_fit = obj_logit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    this_parameter = this_fit.params
    llk0 = obj_logit.loglike(this_parameter.values)

    if (debug == 'Y'):
        print(this_fit.summary())
        print("Model Log-Likelihood Value =", llk0)
        print('\n')

    # Model 1 is yCat = Intercept + xInt
    x = smodel.add_constant(x_int, prepend = True)
    obj_logit = smodel.MNLogit(y, x)
    this_fit = obj_logit.fit(method = 'newton', full_output = True, maxiter = 100, tol = 1e-8)
    this_parameter = this_fit.params
    llk1 = obj_logit.loglike(this_parameter.values)

    if (debug == 'Y'):
        print(this_fit.summary())
        print("Model Log-Likelihood Value =", llk1)

    # Calculate the deviance
    deviance_stat = 2.0 * (llk1 - llk0)
    deviance_df = (len(y.cat.categories) - 1.0)
    deviance_sig = scipy.stats.chi2.sf(deviance_stat, deviance_df)

    mc_fadden_r_sq = 1.0 - (llk1 / llk0)

    return(deviance_stat, deviance_df, deviance_sig, mc_fadden_r_sq)

cat_pred = ['group_size', 'homeowner', 'married_couple']
int_pred = []

test_result = pandas.DataFrame(index = cat_pred + int_pred, columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in cat_pred:
    chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v = chi_square_test(purchase_likelihood[pred], purchase_likelihood['A'], debug = 'Y')
    test_result.loc[pred] = ['Chi-square', chi_sq_stat, chi_sq_df, chi_sq_sig, 'Cramer''V', cramer_v]
    
for pred in int_pred:
    deviance_stat, deviance_df, deviance_sig, mc_fadden_r_sq = deviance_test(purchase_likelihood[pred], purchase_likelihood['A'], debug = 'Y')
    test_result.loc[pred] = ['Deviance', deviance_stat, deviance_df, deviance_sig, 'McFadden''s R^2', mc_fadden_r_sq]

rank_sig = test_result.sort_values('Significance', axis = 0, ascending = True)
print(rank_sig)

rank_assoc = test_result.sort_values('Measure', axis = 0, ascending = False)
print(rank_assoc)


x_category = purchase_likelihood[cat_pred].astype('category')
x_cat = pandas.get_dummies(x_category)
x_data = x_cat.join(purchase_likelihood[int_pred])

y_data = purchase_likelihood['A'].astype('category')

_obj_r_f = ensemble.RandomForestClassifier(criterion = 'entropy', n_estimators = 1000, max_features = 'sqrt',
                                         max_depth = 10, random_state = 27513, bootstrap = True)
this_random_forest = _obj_r_f.fit(x_data, y_data)
this_feature_imp = this_random_forest.feature_importances_

x_freq = x_data.sum(0)

print(this_feature_imp)

# Intercept + group_size + homeowner + REASON * JOB
designX = gr_size
designX = designX.join(xJ)

# Create the columns for the JOB * REASON interaction effect
xRJ = create_interaction(gr_size, xJ)
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