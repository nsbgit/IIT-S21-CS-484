import graphviz
import matplotlib.pyplot as plt
import numpy
import pandas

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree

# Set some options for printing all the columns
pandas.set_option('display.max_columns', None)  
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', -1)
pandas.set_option('precision', 7)

CH = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\Claim_History.csv',
                     delimiter = ',', usecols = ['CLM_COUNT', 'BLUEBOOK', 'CAR_AGE', 'HOMEKIDS', 'KIDSDRIV', 'MVR_PTS', 'TRAVTIME'])

# Set the Claim Flag to 1 if the Number of Claims is positive
CH['CLAIM_FLAG'] = numpy.where(CH['CLM_COUNT'] > 0, 1, 0)

# Drop the missing values
CH = CH.dropna()

# Print number of missing values per variable
print('Number of Missing Values:')
print(pandas.Series.sort_index(CH.isna().sum()))

nObs_all = CH.shape[0]

# Create a 70% Training partition and a 30% Test partition
CH_train, CH_test = train_test_split(CH, test_size = 0.3, random_state = 27513, stratify = CH['CLAIM_FLAG'])

nObs_train = CH_train.shape[0]
nObs_test = CH_test.shape[0]

prob_train = nObs_train / nObs_all
prob_test = nObs_test / nObs_all

print('Partition\t Count\t Poportion')
print('Training\t {:.0f} \t {:.6f}'.format(nObs_train, prob_train))
print('    Test\t {:.0f} \t {:.6f}'.format(nObs_test, prob_test))

# Build a CART model
xName = ['BLUEBOOK', 'CAR_AGE', 'HOMEKIDS', 'KIDSDRIV', 'MVR_PTS', 'TRAVTIME']
xTrain = CH_train[xName]
yTrain = CH_train['CLAIM_FLAG']

objTree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, random_state = 60616)
treeCH = objTree.fit(xTrain, yTrain)

dot_data = tree.export_graphviz(treeCH, out_file = None, impurity = True, filled = True,
                                feature_names = xName, class_names = ['0', '1'])

graph = graphviz.Source(dot_data)
graph

# Score the Test partition
xTest = CH_test[xName]
yTest = CH_test['CLAIM_FLAG']
predProbTest = objTree.predict_proba(xTest)
pCLAIM_FLAG_1 = predProbTest[:,1]

# Question 3(c)
AUC = metrics.roc_auc_score(yTest, pCLAIM_FLAG_1)
print('Area Under Curve: ', AUC)
print('\n')

# Generate the coordinates for the ROC curve
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(yTest, pCLAIM_FLAG_1, pos_label = 1)

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw the ROC curve
plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title('ROC Chart for Test Partition')
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis('equal')
plt.show()

# The Gain and Lift function
def compute_lift_coordinates (
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug = 'N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = numpy.percentile(EventPredProb, numpy.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = numpy.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pandas.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN /totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                    axis = 1, ignore_index = True)
    LiftCoordinates = LiftCoordinates.rename({0:'Decile N',
                                              1:'Decile %',
                                              2:'Gain N',
                                              3:'Gain %',
                                              4:'Response %',
                                              5:'Lift'}, axis = 'columns')

    # Construct the Accumulative Lift chart table
    accCountTable = countTable.cumsum(axis = 0)
    decileN = accCountTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = accCountTable[EventValue]
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    lift = responsePct / overallResponsePct

    accLiftCoordinates = pandas.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                       axis = 1, ignore_index = True)
    accLiftCoordinates = accLiftCoordinates.rename({0:'Acc. Decile N',
                                                    1:'Acc. Decile %',
                                                    2:'Acc. Gain N',
                                                    3:'Acc. Gain %',
                                                    4:'Acc. Response %',
                                                    5:'Acc. Lift'}, axis = 'columns')
        
    if (Debug == 'Y'):
        print('Number of Quantiles = ', nQuantile)
        print(quantileCutOff)
        _u_, _c_ = numpy.unique(quantileIndex, return_counts = True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)

# Get the Lift chart coordinates
lift_coordinates, acc_lift_coordinates = compute_lift_coordinates (
        DepVar = yTest,
        EventValue = 1,
        EventPredProb = pCLAIM_FLAG_1,
        Debug = 'Y')

# Draw the Lift chart
plt.plot(lift_coordinates.index, lift_coordinates['Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title('Lift Chart for Test Partition')
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Lift")
plt.show()

# Draw the Accumulative Lift chart
plt.plot(acc_lift_coordinates.index, acc_lift_coordinates['Acc. Lift'], marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.title('Accumulated Lift Chart for Test Partition')
plt.grid(True)
plt.xticks(numpy.arange(1,11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.show()
