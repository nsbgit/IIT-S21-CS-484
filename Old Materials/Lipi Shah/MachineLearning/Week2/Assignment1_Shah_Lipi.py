# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:58:32 2020

@author: Lipi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

def BinWidth(Interquartile, n):
     bin_width = 2 * Interquartile*(n ** (-1/3))
     return bin_width
 
    
# Driver Function 
if __name__=='__main__': 
    data = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\Week2\\NormalSample.csv',
                       delimiter=',')
    x = data.iloc[:,2].values 
    n = len(x) 
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bin_width =  BinWidth(iqr, n)
    print ("**********************Question 1 *********************************************")
    print ("**********************Answer 1.a *********************************************")
    print("answer 1.a : According to Izenman (1991) method, the recomemded bin width for Histogram x is ", bin_width)
    print ("**********************Answer 1.b *********************************************")
    print("answer 1.b minimum value of the field x is: ",min(x)," and maximum value of the field x is :", max(x))
    a = math.floor(min(x))
    b = math.ceil(max(x))
    print ("**********************Answer 1.c *********************************************")
    print("answer 1.c the largest integer less than the minimum value of the field x is", a , "and smallest integer greater than the maximum value of the field x is", b)
    #question 1.d
    
    print ("**********************Answer 1.d *********************************************")
    h= 0.25
    N = len(x)  
    m = a + h/2
    m_values = []
    p = []
    d = dict()
    b = 36.0
    counter = 0
    while (m <= b):
        m_values.append(m)
        w_u = []
        
        u = (x-m)/h
        for i in u:
            if i> -1/2 and i < 1/2:
                w_u.append(1)
            else:
                w_u.append(0)
        p.append(sum(w_u)/(N * h))
        print("[",m_values[counter],':', p[counter],"]",end = " ")
        counter = counter + 1
        m = m + h
     
    h = 0.25
    num_of_bin = (b-a)/h
    sns.distplot(x, bins=int(num_of_bin),kde=True,color='blue')
    plt.title("Histogram of Density Estimater for attribute x at h = 0.25")
    plt.xlabel("Attribute x values")
    plt.ylabel("Density Estimator")
    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.show()
    
    print ("**********************Answer 1.e *********************************************")
    h= 0.5
    N = len(x)  
    m = a + h/2
    m_values = []
    p = []
    d = dict()
    b = 36.0
    counter = 0
    while (m <= b):
        m_values.append(m)
        w_u = []
        u = (x-m)/h
        for i in u:
            if i> -1/2 and i < 1/2:
                w_u.append(1)
            else:
                w_u.append(0)
        p.append(sum(w_u)/(N * h))
        print("[",m_values[counter],':', p[counter],"]",end = " ")
        counter = counter + 1
        m = m + h
    
    h = 0.5
    num_of_bin = (b-a)/h
    sns.distplot(x, bins=int(num_of_bin),kde=True,color='red')
    plt.title("Histogram of Density Estimater for attribute x at h = 0.5")
    plt.xlabel("Attribute x values")
    plt.ylabel("Density Estimator")
    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.show()

    print ("**********************Answer 1.f *********************************************")
    
    h= 1
    N = len(x)  
    m = a + h/2
    m_values = []
    p = []
    d = dict()
    b = 36.0
    counter = 0
    while (m <= b):
        m_values.append(m)
        w_u = []
        u = (x-m)/h
        for i in u:
            if i> -1/2 and i < 1/2:
                w_u.append(1)
            else:
                w_u.append(0)
        p.append(sum(w_u)/(N * h))
        print("[",m_values[counter],':', p[counter],"]",end = " ")
        counter = counter + 1
        m = m + h

    h = 1
    num_of_bin = (b-a)/h
    sns.distplot(x, bins=int(num_of_bin),kde=True,color='blue')
    plt.title("Histogram of Density Estimater for attribute x at h = 1")
    plt.xlabel("Attribute x values")
    plt.ylabel("Density Estimator")
    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.show()
    
    
    #question 1.g
    print ("**********************Answer 1.g *********************************************")
    h= 2
    N = len(x)  
    m = a + h/2
    m_values = []
    p = []
    d = dict()
    b = 36.0
    counter = 0
    while (m <= b):
        m_values.append(m)
        w_u = []
        u = (x-m)/h
        for i in u:
            if i> -1/2 and i < 1/2:
                w_u.append(1)
            else:
                w_u.append(0)
        p.append(sum(w_u)/(N * h))
        print("[",m_values[counter],':', p[counter],"]",end = " ")
        counter = counter + 1
        m = m + h
        
    h = 2
    num_of_bin = (b-a)/h
    sns.distplot(x, bins=int(num_of_bin),kde=True,color='red')
    plt.title("Histogram of Density Estimater for attribute x at h = 2")
    plt.xlabel("Attribute x values")
    plt.ylabel("Density Estimator")
    plt.grid(axis="x")
    plt.grid(axis="y")
    plt.show()
    
    #####################################question 2 ############################################
    #question 2.a
    print("*******************************Question2*********************************************")
    print ("**********************Answer 2.a *********************************************")
    z = np.percentile(x, [25,50,75])
    lower_wishker = z[0] - (1.5*iqr)
    upper_wishker = z[2] + (1.5*iqr)
    print ("Five Number summery of x as follow :min =", min(x),
           ", Q1 =",z[0],
           ", Q2 =",z[1],
           ", Q3 =",z[2],
           "and max =",max(x))
    print("lower_wishker =", lower_wishker," and upper_wishker =",upper_wishker)
    lower_wishkera=lower_wishker
    upper_wishkera=upper_wishker
    
    #question 2.b
    print ("**********************Answer 2.b *********************************************")
    filter1 = data["group"]==0
    zero=data.where(filter1) 
    x_group0 = zero.iloc[:,2].values 
    x_group0 = [x for x in x_group0 if ~np.isnan(x)]
    
    #summery for group = zero
    z = np.percentile(x_group0, [25,50,75])
    iqr_zero = z[2] - z[0]
    lower_wishker = z[0] - (1.5*iqr_zero)
    upper_wishker = z[2] + (1.5*iqr_zero)
    print ("Five Number summery of x where group = 0 as follow : min =", min(x_group0),
           ", Q1 =",z[0],
           ", Q2 =",z[1],
           ", Q3 =",z[2],
           "and max =",max(x_group0))
    print("Group = 0 : lower_wishker =", lower_wishker," and upper_wishker =",upper_wishker)
    lower_wishkerz=lower_wishker
    upper_wishkerz=upper_wishker
    filter2 = data["group"]== 1
    one=data.where(filter2) 
    x_group1 = one.iloc[:,2].values
    x_group1 = [x for x in x_group1 if ~np.isnan(x)]

    #summery for group = one
    z = np.percentile(x_group1, [25,50,75])
    iqr_one = z[2] - z[0]
    lower_wishker = z[0] - (1.5*iqr_one)
    upper_wishker = z[2] + (1.5*iqr_one)
    print ("Five Number summery of x where group = 1 as follow :min =", min(x_group1),
           ", Q1 =",z[0],
           ", Q2 =",z[1],
           ", Q3 =",z[2],
           "and max = ",max(x_group1))
    print("Group= 1 : lower_wishker =", lower_wishker," and upper_wishker =",upper_wishker)
    lower_wishkero=lower_wishker
    upper_wishkero=upper_wishker
    #question 2.c
    print ("**********************Answer 2.c *********************************************")
    fig = plt.figure()
    fig.suptitle('BoxPlot For X Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(x=data["x"],palette="Set2")
    plt.show()
    #verify the plot
    z = np.percentile(x, [25,50,75])
    iqr_zero = z[2] - z[0]
    lower_wishker = z[0] - (1.5*iqr_zero)
    upper_wishker = z[2] + (1.5*iqr_zero)
    print ("Five Number summery of x  : min =", min(x),
           ", Q1 =",z[0],
           ", Q2 =",z[1],
           ", Q3 =",z[2],
           "and max =",max(x))
    print("lower_wishker =", lower_wishker," and upper_wishker =",upper_wishker)
   
    #question 2.d
    print ("**********************Answer 2.d *********************************************")
    box_data = [data['x'],x_group0, x_group1]
    fig = plt.figure()
    fig.suptitle('BoxPlot For x and x with each group Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=box_data,palette="Set2",orient="h")
    plt.show()
    
    #outlier Detection
    #x Outliers
    print("********OUTLIER DETECTION*********")
    print("-------x Attribute Outliers:------")
    x_list = list(data['x'])
    for i in x_list:
        if i < lower_wishkera:
            print("x_list lower outlier",i)
        elif i > upper_wishkera:
            print("x_list upper outlier",i);
            
    #x_group0 Outliers
    print("----x Attribute Outliers with group = 0-------")
    for i in x_group0:
        if i < lower_wishkerz :
            print("x_Group_0 lower outlier",i)
        elif i > upper_wishkerz:
            print("x_Group_0 upper outlier",i);
    
    #x_group1 Outliers
    print("------x Attribute Outliers group = 1--------")
    for i in x_group1:
        if i < lower_wishkero :
            print("x_Group_1 lower outlier",i)
        elif i > upper_wishkero:
            print("x_Group_1 upper outlier",i);
    
    #***********************************question 3 *******************************************#
    print("*******************************Question3*********************************************")
    fraud = pd.read_csv('C:\\Users\\Lipi\\Downloads\\MachineLearning\Week2\\Fraud.csv',
                       delimiter=',')
    
    #question 3.a
    print ("**********************Answer 3.a *********************************************")
    Fraud_count = fraud[fraud["FRAUD"] == 1].count()['FRAUD']
    Total_case_count = fraud.shape[0]
    fraudulent_perc = (Fraud_count/Total_case_count)*100
    fraudulent_perc = float("{0:.4f}".format(fraudulent_perc))
    print("Percent of investigations are found to be fraudulent = ",fraudulent_perc)
    
    #question 3.b
    print ("**********************Answer 3.b *********************************************")
    #nonfraudulent
    nonfraudulent = fraud[fraud["FRAUD"] == 0]
    #Fraudulent
    fraudulent = fraud[fraud["FRAUD"] == 1]
    
    
    #TOTAL_SPEND Fraudulent - Non Fraudulent Boxplot
    Total_spend_fraudulent = fraudulent["TOTAL_SPEND"]
    Total_spend_nonfraudulent = nonfraudulent["TOTAL_SPEND"]
    Total_spend_both = [Total_spend_nonfraudulent,Total_spend_fraudulent]
    fig = plt.figure()
    plt.xlabel('TOTAL_SPEND', fontsize=10)
    plt.ylabel('FRAUD', fontsize=10)
    fig.suptitle('BoxPlot For TOTAL_SPEND Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=Total_spend_both,palette="Set2",orient="h")
    plt.show()
    
    #DOCTOR_VISITS Fraudulent - Non Fraudulent Boxplot
    DOCTOR_VISITS_fraudulent = fraudulent["DOCTOR_VISITS"]
    DOCTOR_VISITS_nonfraudulent = nonfraudulent["DOCTOR_VISITS"]
    Total_spend_both = [DOCTOR_VISITS_nonfraudulent,DOCTOR_VISITS_fraudulent]
    fig = plt.figure()
    plt.xlabel('DOCTOR_VISITS', fontsize=10)
    plt.ylabel('FRAUD', fontsize=10)
    fig.suptitle('BoxPlot For DOCTOR_VISITS Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=Total_spend_both,palette="Set2",orient="h")
    plt.show()
    
    #NUM_CLAIMS Fraudulent - Non Fraudulent Boxplot
    NUM_CLAIMS_fraudulent = fraudulent["NUM_CLAIMS"]
    NUM_CLAIMS_nonfraudulent = nonfraudulent["NUM_CLAIMS"]
    Total_spend_both = [NUM_CLAIMS_nonfraudulent,NUM_CLAIMS_fraudulent]
    fig = plt.figure()
    plt.xlabel('NUM_CLAIMS', fontsize=10)
    plt.ylabel('FRAUD', fontsize=10)
    fig.suptitle('BoxPlot For NUM_CLAIMS Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=Total_spend_both,palette="Set2",orient="h")
    plt.show()
    
    #MEMBER_DURATION Fraudulent - Non Fraudulent Boxplot
    MEMBER_DURATION_fraudulent = fraudulent["MEMBER_DURATION"]
    MEMBER_DURATION_nonfraudulent = nonfraudulent["MEMBER_DURATION"]
    MEMBER_DURATION_nonfraudulent_both = [MEMBER_DURATION_nonfraudulent,MEMBER_DURATION_fraudulent]
    fig = plt.figure()
    plt.xlabel('MEMBER_DURATION', fontsize=10)
    plt.ylabel('FRAUD', fontsize=10)
    fig.suptitle('BoxPlot For MEMBER_DURATION Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=MEMBER_DURATION_nonfraudulent_both,palette="Set2",orient="h")
    plt.show()
    
    
    #OPTOM_PRESC Fraudulent - Non Fraudulent Boxplot
    OPTOM_PRESC_fraudulent = fraudulent["OPTOM_PRESC"]
    OPTOM_PRESC_nonfraudulent = nonfraudulent["OPTOM_PRESC"]
    OPTOM_PRESC_nonfraudulent_both = [OPTOM_PRESC_nonfraudulent,OPTOM_PRESC_fraudulent]
    fig = plt.figure()
    plt.xlabel('OPTOM_PRESC', fontsize=10)
    plt.ylabel('FRAUD', fontsize=10)
    fig.suptitle('BoxPlot For OPTOM_PRESC Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=OPTOM_PRESC_nonfraudulent_both,palette="Set2",orient="h")
    plt.show()
    
    #NUM_MEMBERS Fraudulent - Non Fraudulent Boxplot
    NUM_MEMBERS_fraudulent = fraudulent["NUM_MEMBERS"]
    NUM_MEMBERS_nonfraudulent = nonfraudulent["NUM_MEMBERS"]
    NUM_MEMBERS_nonfraudulent_both = [NUM_MEMBERS_nonfraudulent,NUM_MEMBERS_fraudulent]
    fig = plt.figure()
    plt.xlabel('NUM_MEMBERS', fontsize=10)
    plt.ylabel('FRAUD', fontsize=10)
    fig.suptitle('BoxPlot For NUM_MEMBERS Attribute', fontsize=14, fontweight='bold')
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=NUM_MEMBERS_nonfraudulent_both,palette="Set2",orient="h")
    plt.show()
    
    print ("**********************Answer 3.c *********************************************")
    #question 3.c (i)
    import numpy as np
    from numpy import linalg as LA
    from scipy import linalg as LA2
    fraud_nn = fraud.iloc[:,[2,3,4,5,6,7]] 
    fraud_nn= np.matrix(fraud_nn)
    #print(fraud_nn.transpose())
    xtx = fraud_nn.transpose() * fraud_nn
    #print(xtx)
    evals, evecs = LA.eigh(xtx)
    print("Eigenvalues of fraud_nn matrix = \n", evals)
    
    #question 3.c (ii)
    #transformation of matrix
    print("**********************************************")
    transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
    print("Transformation Matrix transf= \n", transf)
    
    # Here is the transformed 
    transf_fraud_nn = fraud_nn * transf;
    print("The Transformed transf_fraud_nn = \n", transf_fraud_nn)
    
    xtx = transf_fraud_nn.transpose() * transf_fraud_nn;
    print("Identity Matrix = \n", xtx)
    
    
    
    #question 3.d
    print ("**********************Answer 3.d *********************************************")
    from sklearn.neighbors import KNeighborsClassifier
    trainData = fraud_nn
    target = fraud['FRAUD']
    
    #apply KNeighborsClassifier : Build the model and compare with actaul target value
    neigh = KNeighborsClassifier(n_neighbors=5 , metric = 'euclidean', algorithm = 'brute')
    nbrs = neigh.fit(trainData, target)
    #how much cases give percentage how much percentage correctly classified wrt to target
    score = nbrs.score(trainData, target) 
    #defines that how much accuracy can be achieve by defined model
    print("score function value = ",score) 
    missclassification = (1 -score)*100
    
    
    #question 3.e
    print ("**********************Answer 3.e *********************************************")
    import pandas
    from sklearn.neighbors import NearestNeighbors as kNN
    focal_input = [7500,15,3,127,2,2]
    
    #KNN with 5 neighbor
    kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
    #transform the focal_Input
    transf_focal_input = focal_input * transf;
    
    nbrs_t = kNNSpec.fit(transf_fraud_nn)
    distances, indices = nbrs_t.kneighbors(transf_fraud_nn)
    
    #N neighbors
    Target_neighbors = nbrs_t.kneighbors(transf_focal_input, return_distance = False)
    print("Target_Input's Neighbors = \n", Target_neighbors)
    print(fraud.iloc[list(Target_neighbors[0])])
    #question 3.f
    print ("**********************Answer 3.f *********************************************")
    print("Predicted Probability of Fraudulent ",nbrs.predict(transf_fraud_nn))
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
   
   
   
    
    
     
    
    