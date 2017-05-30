import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas

#load data into dataframes (Vectors)
X = []
Y = []
X = pandas.read_csv('data_1d.csv', header=None, usecols=[0], squeeze=True)
Y = pandas.read_csv('data_1d.csv', header=None, usecols=[1], squeeze=True)

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# find a and b for line of best fit (Yhat = ax + b)

denonimator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denonimator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denonimator

# calculate predicted Y
Yhat = a * X + b

# plot line of best fit
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# check our work with R-squared
# Rsquared = 1 - Sum of square residuals / Sum of square total
# SSresiduals = Sum((yi - yhat_i)^2)
# SStotal = Sum((yi - y_mean)^2)
# if SSresiduals ~ 0 then Rsquared is ~1 thus perfect model
# if Rsquared = 0 then SSres/SStotal =1, bad model as we just have the average of y
# if Rsquared < 0 then model is worse than using mean, recheck model
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The r-quared is: ", r2)

