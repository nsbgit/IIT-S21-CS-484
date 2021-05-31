import pandas as pd
import numpy
import scipy
from sklearn import naive_bayes

def row_with_column (rowVar, columnVar, show = 'ROW'):
    countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
    print("Frequency Table: \n", countTable)
    print( )
    return

purchase_ll = pd.read_csv('Purchase_Likelihood.csv')
purchase_ll = purchase_ll.dropna()

cat_pred = ['group_size', 'homeowner', 'married_couple']

print("Q3.a)(5 points) Show in a table the frequency counts and the Class Probabilities of the target variable.")
freq = purchase_ll.groupby('insurance').size()
table = pd.DataFrame(columns = ['Frequency count', 'class probability'])
table['Frequency count'] = freq
table['class probability'] = table['Frequency count']/purchase_ll.shape[0]
print(table)
print("*"*50)

print("Q3.b)(5 points) Show the crosstabulation table of the target variable by the feature group_size.  The table contains the frequency counts.")
row_with_column(purchase_ll['insurance'],purchase_ll['group_size'],'ROW')
print("*"*50)

print("Q3.c)(5 points) Show the crosstabulation table of the target variable by the feature homeowner.  The table contains the frequency counts.")
row_with_column(purchase_ll['insurance'],purchase_ll['homeowner'],'ROW')
print("*"*50)

print("Q3.d)(5 points) Show the crosstabulation table of the target variable by the feature married_couple.  The table contains the frequency counts.")
row_with_column(purchase_ll['insurance'],purchase_ll['married_couple'],'ROW')
print("*"*50)

# Define a function that performs the Chi-square test
def ChiSquareTest(xCat, yCat, debug='N'):
    obsCount = pd.crosstab(index=xCat, columns=yCat, margins=False, dropna=True)
    cTotal = obsCount.sum(axis=1)
    rTotal = obsCount.sum(axis=0)
    nTotal = numpy.sum(rTotal)
    expCount = numpy.outer(cTotal, (rTotal / nTotal))

    if (debug == 'Y'):
        print('Observed Count:\n', obsCount)
        print('Column Total:\n', cTotal)
        print('Row Total:\n', rTotal)
        print('Overall Total:\n', nTotal)
        print('Expected Count:\n', expCount)
        print('\n')

    chiSqStat = ((obsCount - expCount) ** 2 / expCount).to_numpy().sum()
    chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
    chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = numpy.sqrt(cramerV)

    return (chiSqStat, chiSqDf, chiSqSig, cramerV)

print("Q3.e)(5 points) Calculate the Cramer’s V statistics for the above three crosstabulations tables.  Based on these Cramer’s V statistics, which feature has the largest association with the target insurance?")
test_result = pd.DataFrame(index = cat_pred, columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])

for pred in cat_pred:
    chi_sq_stat, chi_sq_df, chi_sq_sig, cramer_v = ChiSquareTest(purchase_ll[pred], purchase_ll['insurance'], debug = 'N')
    test_result.loc[pred] = ['Chi-square', chi_sq_stat, chi_sq_df, chi_sq_sig, 'Cramer''V', cramer_v]

rank_assoc = test_result.sort_values('Measure', axis = 0, ascending = False)
print(rank_assoc)
print("*"*50)

print("Q3.f)(10 points) For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for insurance = 0, 1, 2 based on the Naïve Bayes model.  List your answers in a table with proper labeling.")
xTrain = purchase_ll[cat_pred].astype('category')
yTrain = purchase_ll['insurance'].astype('category')

_objNB = naive_bayes.MultinomialNB(alpha = 1.0e-10)
thisFit = _objNB.fit(xTrain, yTrain)
gs_d = [1, 2, 3, 4]
ho_d = [0, 1]
mc_d = [0, 1]
insurance_d = [0, 1, 2]

x_data = []

for gsd in gs_d:
    for hod in ho_d:
        for mcd in mc_d:
            data = [gsd, hod, mcd]
            x_data = x_data + [data]

x_test = pd.DataFrame(x_data, columns=['group_size', 'homeowner', 'married_couple'])
x_test = x_test[cat_pred].astype('category')
y_test_pred_prob = pd.DataFrame(_objNB.predict_proba(x_test), columns=['p_in_0', 'p_in_1', 'p_in_2'])
y_test_score = pd.concat([x_test, y_test_pred_prob], axis=1)
print(y_test_score)
print("*"*50)

print("Q3.g)(5 points) Based on your model, what value combination of group_size, homeowner, and married_couple will maximize the odds value Prob(insurance = 1) / Prob(insurance = 0)?  What is that maximum odd value?")
y_test_score['odd value(p_in_1/p_in_0)'] = y_test_score['p_in_1'] / y_test_score['p_in_0']
print(y_test_score[['group_size','homeowner','married_couple','odd value(p_in_1/p_in_0)']])
print(y_test_score.loc[y_test_score['odd value(p_in_1/p_in_0)'].idxmax()])