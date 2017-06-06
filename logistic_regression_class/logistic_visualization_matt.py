import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

# THIS ONLY WORKS FOR BAYES blah blah datasets, not general enough
# go look at logistic_3_matt.py for gradient descent general purpose
# solution

# create two gaussian clouds

# make first 50 points centrer around (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))

# make next 50 points center around (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# bias column(?)
Xb = np.concatenate((np.ones((N, 1)), X), axis=1)

# first half of targets category 0...
T = np.array([0] * 50 + [1] * 50)

# use closed-form solution for logistic regression
w = np.array([0, 4, 4])
Y = expit(Xb.dot(w))

# y = -x

plt.scatter(X[:, 0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()


def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


print('Cross entropy error is:', calculate_cross_entropy(T, Y))
