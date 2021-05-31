import matplotlib.pyplot as plt
import numpy
import pandas

numpy.random.seed(seed = 20191106)

# Generate the first multivariate normal sample
n0 = 40
m0 = [1.0, 1.0]
s0 = [0.8, 0.8]
corr0 = [[1.0, -0.5], [-0.5, 1.0]]

cov0 = corr0 * numpy.outer(s0, s0)
print('Covariance Matrix 0:\n', cov0)

mvn0 = numpy.append(numpy.full((n0,1), 0),
                    numpy.random.multivariate_normal(m0, cov0, n0),
                    axis = 1)

# Generate the second multivariate normal sample
n1 = 60
m1 = [-1.0, -1.0]
s1 = [0.8, 0.8]
corr1 = [[1.0, -0.3], [-0.3, 1.0]]

cov1 = corr1 * numpy.outer(s1, s1)
print('Covariance Matrix 1:\n', cov1)

mvn1 = numpy.append(numpy.full((n1,1), 1),
                    numpy.random.multivariate_normal(m1, cov1, n1),
                    axis = 1)

mvnData = numpy.append(mvn0, mvn1, axis = 0)

# Scatterplot without any prior knowledge of the grouping variable
plt.figure(figsize=(10,10))
plt.scatter(x = mvnData[:,1], y = mvnData[:,2], c = 'blue', marker = 'o', s = 50)
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

with pandas.ExcelWriter('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\MVN100.xlsx') as writer:
    pandas.DataFrame(mvnData, columns = ['Group', 'X','Y']).to_excel(writer, sheet_name = 'MVN100', index = False)
