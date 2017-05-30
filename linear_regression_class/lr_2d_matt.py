# 1d Yhat = a*X + b
# 2d Yhat = w.T*X + b
# w and x have to be same size
# x i NxD, N samples, D features
# X = 0 0 1   w = 1
#     0 1 0       2
#     1 0 0       3
#     1 1 0
# X is 4x3 and w is 3x1
# output is a 4x1 matrix
# y1 = w.T*x1 = [1 2 3] [0] = 3 = x1.T*w
#                       [0]
#                       [1]
# solve for w
# w = (X.T*X)^-1 * X.T*y
# w = np.linalg.solve(X.T*X,X.T*y)
# 
#
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as Axes3D

# load the data
'''
X = []
Y = []
for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1]) # add the bias term
    Y.append(float(y))
X = np.array(X)
Y = np.array(Y)
'''

X = pd.read_csv('data_2d.csv', header=None, usecols=[0,1])
X = np.array(X)
X = np.insert(X, 0, 1, axis=1) # insert bias row
Y = pd.read_csv('data_2d.csv', header=None, usecols=[2], squeeze=True)
Y = np.array(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show(fig)

# calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w) # different than 1d regression

# compute r-squared (same as 1d regression)
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print('r-squared:', r2)
