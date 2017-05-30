import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# rows are each patient's data
# columns are features
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds

dataframe = pd.read_excel('mlr02.xls', encoding_override='ascii')
X = dataframe.as_matrix()

# goal is to predict X1 based on X2,X3
plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

dataframe['ones'] = 1
Y = dataframe['X1']
X = dataframe[['X2', 'X3', 'ones']]
X2only = dataframe[['X2', 'ones']]
X3only = dataframe[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1) / d2.dot(d2)
    return r2


print("r2 for x2 only:", get_r2(X2only, Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for both:", get_r2(X, Y))

# add another feature that is just noise
print("Adding noise feature...")
dataframe['noise'] = pd.Series(np.random.randint(0, 100, size=(len(dataframe.index))))
X = dataframe[['X2', 'X3', 'noise', 'ones']]
Xnoise = dataframe[['noise', 'ones']]
print("r2 for x2 only:", get_r2(X2only, Y))
print("r2 for x3 only:", get_r2(X3only, Y))
print("r2 for noise only:", get_r2(Xnoise, Y))
print("r2 for all:", get_r2(X, Y))