import numpy
import pandas
import sklearn.cluster as cluster

cars = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\cars.csv',delimiter=',')

trainData = cars[["EngineSize", "Cylinders", "Horsepower", "MPG_City", "MPG_Highway", "Weight", "Wheelbase", "Length"]]
trainData = trainData.dropna(axis='index')

# Check number of missing values in each column
trainData.isnull().sum()

myCorr = numpy.corrcoef(trainData, rowvar = False)

objCluster = cluster.SpectralBiclustering(n_clusters=(3,3), random_state=27513)
myCluster = objCluster.fit(myCorr)

print(myCluster.row_labels_)
print(myCluster.column_labels_)

# Show the variance names in each cluster
cID = pandas.DataFrame(myCluster.column_labels_, columns = ["Cluster ID"])

vName = pandas.DataFrame(trainData.columns, columns = ["Varname"])

vNameCluster = pandas.concat([cID, vName], axis = 1)

print(vNameCluster)