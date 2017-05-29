import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
X = pd.read_csv('data_poly.csv', header=None, usecols=[0])
X = np.array(X, dtype='float')
X = np.append(X, X * X, axis=1)  # add squared values
X = np.insert(X, 0, 1, axis=1)   # add bias column

Y = pd.read_csv('data_poly.csv', header=None, usecols=[1], squeeze=True)
Y = np.array(Y, dtype='float')

# plt.scatter(X[:, 1], Y)
# plt.show()

# calculate weights
# polynomial is multiple linear regression but input table X is different (squares of X)
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# plot it all together
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

# calculate r-quared
d1 = Y - Yhat
d2 = Y - Y.mean()
r_squared = 1 - d1.dot(d1) / d2.dot(d2)
print('r-squred is:', r_squared)
