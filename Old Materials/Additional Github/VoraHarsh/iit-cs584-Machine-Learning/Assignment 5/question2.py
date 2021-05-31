import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import sklearn.svm as svm
import matplotlib.pyplot as plt

spiral_df = pd.read_csv('SpiralWithCluster.csv', delimiter=',', usecols=['x', 'y', 'SpectralCluster'])

print("Q2.a)(5 points) What is the equation of the separating hyperplane?  Please state the coefficients up to seven decimal places.")

x_Train = spiral_df[['x','y']]
y_Train = spiral_df['SpectralCluster']

svm_model = svm.SVC(kernel='linear', random_state=20200408, max_iter=-1, decision_function_shape='ovr')
thisFit = svm_model.fit(x_Train, y_Train)
y_predictClass = thisFit.predict(x_Train)

spiral_df['_PredictedClass_'] = y_predictClass

print('Intercept = ', np.round(thisFit.intercept_,7))
print('Coefficients = ', np.round(thisFit.coef_,7))

print(
    f'Equation of the separating hyperplane is "𝑤_0+𝐰_1*𝐱+w_2*y=𝟎"  ==> '
    f'({np.round(thisFit.intercept_[0], 7)}) '
    f'+ ({np.round(thisFit.coef_[0][0], 7)}*x) '
    f'+ ({np.round(thisFit.coef_[0][1], 7)}*y) = 𝟎')
print("")

print("Q2.b)(5 points) What is the misclassification rate?")
misclassification  = 1 - metrics.accuracy_score(y_Train, y_predictClass)
print("Misclassification rate is ", misclassification)
print("")

print("Q2.c)(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.")

w = thisFit.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (thisFit.intercept_[0]) / w[1]

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subspiral_df = spiral_df[spiral_df['_PredictedClass_'] == i]
    plt.scatter(x = subspiral_df['x'], y = subspiral_df['y'], c = carray[i], label = i, s = 25)
plt.plot(xx, yy, color = 'black', linestyle = 'dotted')
plt.grid(True)
plt.title('SVM Scatter Plot for Predicted Class with Hyperplane')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
print("")

print("Q2.d)(10 points) Please express the spiral_df as polar coordinates.  Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster variable (0 = Red and 1 = Blue).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.")

spiral_df['radius'] = np.sqrt(spiral_df['x']**2 + spiral_df['y']**2)
spiral_df['theta'] = np.arctan2(spiral_df['y'], spiral_df['x'])

def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)

spiral_df['theta'] = spiral_df['theta'].apply(customArcTan)

x_Train = spiral_df[['radius','theta']]
y_Train = spiral_df['SpectralCluster']

carray = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    subspiral_df = spiral_df[spiral_df['SpectralCluster'] == i]
    plt.scatter(x=subspiral_df['radius'], y=subspiral_df['theta'], c=carray[i], label=i)
plt.grid(True)
plt.title('SVM Scatter Plot for Polar Coordinates')
plt.xlabel('Radius')
plt.ylabel('Theta')
plt.ylim(-1, 7)
plt.legend(title = 'Spectral Cluster', loc = 'lower right')
plt.show()
print("")

print("Q2.e)(10 points) You should expect to see three distinct strips of points and a lone point.  Since the SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of the chart in (d), values 1, 2,and 3 for the next three strips of points.")
print("Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.")

group = np.zeros(spiral_df.shape[0])
spiral_df['Group'] = 2

spiral_df.loc[(spiral_df['theta'] > 3.00) & (spiral_df['radius'] < 2.5), 'Group'] = 1
spiral_df.loc[(spiral_df['theta'] > 5.00) & (spiral_df['radius'] < 3.0), 'Group'] = 1
spiral_df.loc[(spiral_df['theta'] > 6.00) & (spiral_df['radius'] < 1.5), 'Group'] = 0
spiral_df.loc[(spiral_df['theta'] <= 2.00) & (spiral_df['radius'] > 2.5), 'Group'] = 3
spiral_df.loc[(spiral_df['theta'] < 3.20) & (spiral_df['radius'] > 3.0), 'Group'] = 3

# plot coordinates divided into four group
color_array = ['red', 'blue', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    x_y = spiral_df[spiral_df['Group'] == i]
    plt.scatter(x=x_y['radius'], y=x_y['theta'], c=color_array[i], label=i)
plt.xlabel('Radius-Coordinates')
plt.ylabel('Theta-Coordinates')
plt.title('SVM Polar Coordinates Scatter Plot for 4 Groups')
plt.ylim(-1, 7)
plt.legend(title='Group', loc='best')
plt.grid(True)
plt.show()
print("")

print("Q2.f)(10 points) Since the graph in (e) has four clearly separable and neighboring segments, we will apply the Support Vector Machine algorithm in a different way.  Instead of applying SVM once on a multi-class target variable, you will SVM three times, each on a binary target variable.")
print("SVM 0: Group 0 versus Group 1")
print("SVM 1: Group 1 versus Group 2")
print("SVM 2: Group 2 versus Group 3")
print("Please give the equations of the three hyperplanes.")

# build SVM 0: Group 0 versus Group 1
spiral01 = spiral_df[spiral_df['Group'].isin([0,1])]
train_subset0 = spiral01[['radius', 'theta']]
y_train0 = spiral01['Group']
svm_1 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
thisFit01 = svm_1.fit(train_subset0,y_train0)

print(
    f'Equation of the separating hyperplane for SVM 0 is "𝑤_0+𝐰_1*𝐱+w_2*y=𝟎"  ==> '
    f'({np.round(thisFit01.intercept_[0] ,7)})'
    f' + ({np.round(thisFit01.coef_[0][0], 7)}*x)'
    f' + ({np.round(thisFit01.coef_[0][1], 7)}*y) = 𝟎')

w = thisFit01.coef_[0]
a = -w[0] / w[1]
xx1 = np.linspace(1, 4)
yy1 = a * xx1 - (thisFit01.intercept_[0]) / w[1]

# build SVM 0: Group 1 versus Group 2
spiral12 = spiral_df[spiral_df['Group'].isin([1,2])]
train_subset1 = spiral12[['radius', 'theta']]
y_train1 = spiral12['Group']
svm_2 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
thisFit12 = svm_2.fit(train_subset1,y_train1)

print(
    f'Equation of the separating hyperplane for SVM 1 is "𝑤_0+𝐰_1*𝐱+w_2*y=𝟎"  ==> '
    f'({np.round(thisFit12.intercept_[0] ,7)})'
    f' + ({np.round(thisFit12.coef_[0][0], 7)}*x)'
    f' + ({np.round(thisFit12.coef_[0][1], 7)}*y) = 𝟎')

w = thisFit12.coef_[0]
a = -w[0] / w[1]
xx2 = np.linspace(1, 4)
yy2 = a * xx2 - (thisFit12.intercept_[0]) / w[1]

# build SVM 0: Group 2 versus Group 3
spiral23 = spiral_df[spiral_df['Group'].isin([2,3])]
train_subset2 = spiral23[['radius', 'theta']]
y_train2 = spiral23['Group']
svm_3 = svm.SVC(kernel="linear", random_state=20200408, decision_function_shape='ovr', max_iter=-1)
thisFit23 = svm_3.fit(train_subset2,y_train2)

print(
    f'Equation of the separating hyperplane for SVM 2 is "𝑤_0+𝐰_1*𝐱+w_2*y=𝟎"  ==> '
    f'({np.round(thisFit23.intercept_[0] ,7)})'
    f' + ({np.round(thisFit23.coef_[0][0], 7)}*x)'
    f' + ({np.round(thisFit23.coef_[0][1], 7)}*y) = 𝟎')

w = thisFit23.coef_[0]
a = -w[0] / w[1]
xx3 = np.linspace(1, 4)
yy3 = a * xx3 - (thisFit23.intercept_[0]) / w[1]
print("")

print("Q2.g)(5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.")

# plot polar coordinates and hyperplanes
plt.figure(figsize=(10,10))
for i in range(4):
    x_y = spiral_df[spiral_df['Group'] == i]
    plt.scatter(x_y['radius'], x_y['theta'], c=color_array[i], label=i)
plt.plot(xx1, yy1, color='black', linestyle='dotted')
plt.plot(xx2, yy2, color='black', linestyle='dotted')
plt.plot(xx3, yy3, color='black', linestyle='dotted')
plt.xlabel('Radius')
plt.title('SVM Polar Coordinates Scatter Plot and Hyperplanes for 4 Groups')
plt.ylabel('Theta')
plt.legend(title='Group', loc='best')
plt.grid(True)
plt.show()
print("")

print("Q2.h)(10 points) Convert the observations along with the hyperplanes from the polar coordinates back to the Cartesian coordinates. Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster (0 = Red and 1 = Blue). Besides, plot the hyper-curves as dotted lines to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.")
print("Based on your graph, which hypercurve do you think is not needed?")

#Polar coordinates to the Cartesian coordinates
h1_xx1 = xx1 * np.cos(yy1)
h1_yy1 = xx1 * np.sin(yy1)
h2_xx2 = xx2 * np.cos(yy2)
h2_yy2 = xx2 * np.sin(yy2)
h3_xx3 = xx3 * np.cos(yy3)
h3_yy3 = xx3 * np.sin(yy3)

color_array = ['red', 'blue']
plt.figure(figsize=(10,10))
for i in range(2):
    x_y = spiral_df[spiral_df['SpectralCluster'] == i]
    plt.scatter(x_y['x'], x_y['y'], c=color_array[i], label=i)
plt.plot(h1_xx1, h1_yy1, color='green', linestyle='dotted')
plt.plot(h2_xx2, h2_yy2, color='black', linestyle='dotted')
plt.plot(h3_xx3, h3_yy3, color='black', linestyle='dotted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(title='Spectral_Cluster', loc='best')
plt.title('SVM Cartesian Coordinates Scatter Plot and Hypercurves')
plt.grid(True)
plt.show()
print("")
