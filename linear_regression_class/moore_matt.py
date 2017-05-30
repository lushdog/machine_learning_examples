import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

STRIP_REFERENCE = re.compile(r'\[\d+\]')
NON_DECIMAL = re.compile(r'[^\d]')

Z = pd.read_csv('moore.csv', delimiter=r'\t', usecols=[1, 2], engine='python', header=None)
X, Y = [np.array([NON_DECIMAL.sub('', STRIP_REFERENCE.sub('', e)) for e in Z[i]], dtype='float') for i in [2, 1]]
print(X)
print(Y)
plt.scatter(X, Y) 
plt.show()

Y = np.log(Y)  # exponential to linear
plt.scatter(X, Y)
plt.show()

#manually calculate the line of best fit
denominator = X.dot(X) - X.mean() * X.sum()
a = ((X.dot(Y) - Y.mean() * X.sum()) / denominator)
b = ((Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator)
Yhat = a * X + b
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("a:", a, "b:", b, "r-squared:", r2)

# bonus marks :)
# numpy.poylyfit() does the least squares line of best fit as above
Z = np.polyfit(X, Y, 1)
a1 = Z[0] # slope
b1 = Z[1] # intercept
Yhat1 = a1 * X + b1
print("a1", a1, "b1:", b1)

# scipy.stats.linregress() does the same
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
print("a2:", slope, "b2:", intercept, "r2_2:", r_value * r_value)

# how to calculate time to double
# log(tc) = a*year + b
# tc = exp(b) * exp(a * year)
# 2*tc = 2 * exp(b) * exp(a * year) = exp(ln(2)) * exp(b) * exp(a * year)
#       = exp(b) * exp(a * year + ln(2))
# exp(b)*exp(a*year2) = exp(b) * exp(a * year1 + ln2)
# a * year2 = a * year1 + ln2
# year2 = year1 + ln2a
print("time to double:", np.log(2) / a, "in years.")
