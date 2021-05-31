import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.decomposition as decomposition

# Initialize the random seed value
numpy.random.seed(1017)

# Specify two independent random variables
z0 = numpy.random.normal(loc = 0.0, scale = 1.0, size = 5000)

z1 = numpy.random.uniform(low = -numpy.sqrt(3), high = numpy.sqrt(3), size = 5000)

# Specify two dependent random variables
z2 = (z0 + z1) / numpy.sqrt(2)

z3 = (z0 - z1) / numpy.sqrt(2)

z = pandas.DataFrame()

z['z0'] = z0
z['z1'] = z1
z['z2'] = z2
z['z3'] = z3

print('Descriptive Statistics: \n', z.describe())

# Calculate the Correlations among the variables
zCorrelation = z.corr(method = 'pearson', min_periods = 1)
print('Empirical Correlation: \n', zCorrelation)

# Eigenvalue decomposition
evals, evecs = numpy.linalg.eigh(zCorrelation)

print("Eigenvalues of zCorrelation = \n", evals)
print("Eigenvectors of zCorrelation = \n",evecs)

# Extract the Principal Components
myPCA = decomposition.PCA(n_components = 4, svd_solver = 'full')
myPCA.fit(z)

print("Mean: \n", myPCA.mean_)
print('Explained Variance: \n', myPCA.explained_variance_)
print('Explained Variance Ratio: \n', myPCA.explained_variance_ratio_)
print('Principal Components: \n', myPCA.components_)
print(numpy.transpose( myPCA.components_))

plt.plot(myPCA.explained_variance_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.xticks(numpy.arange(0,4))
plt.grid(True)
plt.show()




print(myPCA.singular_values_)

z_transformed = myPCA.fit_transform(z)
numpy.linalg.norm(z_transformed[:,0])
numpy.linalg.norm(z_transformed[:,1])
numpy.linalg.norm(z_transformed[:,2])
numpy.linalg.norm(z_transformed[:,3])