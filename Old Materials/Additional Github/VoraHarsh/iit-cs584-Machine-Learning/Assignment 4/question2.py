import pandas as pd
import statsmodels.api as stats

def create_interaction(in_df1, in_df2):
    name1 = in_df1.columns
    name2 = in_df2.columns
    out_df = pd.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            out_df[outName] = in_df1[col1] * in_df2[col2]
    return (out_df)

purchase_ll = pd.read_csv('Purchase_Likelihood.csv')
purchase_ll = purchase_ll.dropna()

# Specify Origin as a categorical variable
y = purchase_ll['insurance'].astype('category')

# Specify GROUP_SIZE, HOMEOWNER and MARRIED_COUPLE as categorical variables
xgs = pd.get_dummies(purchase_ll[['group_size']].astype('category'))
xho = pd.get_dummies(purchase_ll[['homeowner']].astype('category'))
xmc = pd.get_dummies(purchase_ll[['married_couple']].astype('category'))

designX = xgs
designX = designX.join(xho)
designX = designX.join(xmc)
# Create the columns for the GROUP_SIZE * HOMEOWNER interaction effect
xgsho = create_interaction(xgs, xho)
designX = designX.join(xgsho)
designX = stats.add_constant(designX, prepend=True)
# Create the columns for the GROUP_SIZE * MARRIED_COUPLE interaction effect
xgsmc = create_interaction(xgs, xmc)
designX = designX.join(xgsmc)
designX = stats.add_constant(designX, prepend=True)
# Create the columns for the HOMEOWNER * MARRIED_COUPLE interaction effect
xhomc = create_interaction(xho, xmc)
designX = designX.join(xhomc)
designX = stats.add_constant(designX, prepend=True)

logit = stats.MNLogit(y, designX)
this_fit = logit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)

print("Q2.a)(10 points) For each of the sixteen possible value combinations of the three features, calculate the predicted probabilities for insurance = 0, 1, 2 based on your multinomial logistic model.  List your answers in a table with proper labeling.")
gs_unique = [1,2,3,4]
ho_unique = [0,1]
mc_unique = [0,1]

x_data = []

for gsunique in gs_unique:
    for hounique in ho_unique:
        for mcunique in mc_unique:
            data = [gsunique,hounique,mcunique]
            x_data = x_data + [data]

x_input = pd.DataFrame(x_data, columns=['group_size','homeowner','married_couple'])
xgs = pd.get_dummies(x_input[['group_size']].astype('category'))
xho = pd.get_dummies(x_input[['homeowner']].astype('category'))
xmc = pd.get_dummies(x_input[['married_couple']].astype('category'))
designX = xgs
designX = designX.join(xho)
designX = designX.join(xmc)
# Create the columns for the GROUP_SIZE * HOMEOWNER interaction effect
xgsho = create_interaction(xgs, xho)
designX = designX.join(xgsho)
designX = stats.add_constant(designX, prepend=True)
# Create the columns for the GROUP_SIZE * MARRIED_COUPLE interaction effect
xgsmc = create_interaction(xgs, xmc)
designX = designX.join(xgsmc)
designX = stats.add_constant(designX, prepend=True)
# Create the columns for the HOMEOWNER * MARRIED_COUPLE interaction effect
xhomc = create_interaction(xho, xmc)
designX = designX.join(xhomc)
designX = stats.add_constant(designX, prepend=True)
insurance_pred = this_fit.predict(exog = designX)
insurance_output = pd.concat([x_input, insurance_pred],axis=1)
print(insurance_output)

print("Q2.b)(5 points) Based on your answers in (a), what value combination of group_size, homeowner, and married_couple will maximize the odds value Prob(insurance = 1) / Prob(insurance = 0)?  What is that maximum odd value?")
insurance_output['odd_value(p_in_1/p_in_0)'] = insurance_output[1] / insurance_output[0]
print(insurance_output[['group_size','homeowner','married_couple','odd_value(p_in_1/p_in_0)']])
print(insurance_output.loc[insurance_output['odd_value(p_in_1/p_in_0)'].idxmax()])

print("Q2.c)(5 points) Based on your model, what is the odds ratio for group_size = 3 versus group_size = 1, and insurance = 2 versus insurance = 0?(Hint: The odds ratio is this odds (Prob(insurance = 2) / Prob(insurance = 0) | group_size = 3) divided by this odds ((Prob(insurance = 2) / Prob(insurance = 0) | group_size = 1).))")
prob_in_2_gs_3 = (purchase_ll[purchase_ll['group_size']==3].groupby('insurance').size()[2]/purchase_ll[purchase_ll['group_size']==3].shape[0])
prob_in_0_gs_3 = (purchase_ll[purchase_ll['group_size']==3].groupby('insurance').size()[0]/purchase_ll[purchase_ll['group_size']==3].shape[0])
r1 = prob_in_2_gs_3/prob_in_0_gs_3

prob_in_2_gs_1 = (purchase_ll[purchase_ll['group_size']==1].groupby('insurance').size()[2]/purchase_ll[purchase_ll['group_size']==1].shape[0])
prob_in_0_gs_1 = (purchase_ll[purchase_ll['group_size']==1].groupby('insurance').size()[0]/purchase_ll[purchase_ll['group_size']==1].shape[0])
r2 = prob_in_2_gs_1/prob_in_0_gs_1
r = r1/r2
print(r)

print("Q2.d)(5 points) Based on your model, what is the odds ratio for homeowner = 1 versus homeowner = 0, and insurance = 0 versus insurance = 1?")
prob_in_0_ho_1 = (purchase_ll[purchase_ll['homeowner']==1].groupby('insurance').size()[0]/purchase_ll[purchase_ll['homeowner']==1].shape[0])
prob_in_1_ho_1 = (purchase_ll[purchase_ll['homeowner']==1].groupby('insurance').size()[1]/purchase_ll[purchase_ll['homeowner']==1].shape[0])
r1 = prob_in_0_ho_1/prob_in_1_ho_1

prob_in_0_ho_0 = (purchase_ll[purchase_ll['homeowner']==0].groupby('insurance').size()[0]/purchase_ll[purchase_ll['homeowner']==0].shape[0])
prob_in_1_ho_0 = (purchase_ll[purchase_ll['homeowner']==0].groupby('insurance').size()[1]/purchase_ll[purchase_ll['homeowner']==0].shape[0])
r2 = prob_in_0_ho_0/prob_in_1_ho_0
r = r1/r2
print(r)