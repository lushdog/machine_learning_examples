import numpy as np
from scipy.special import expit

N = 100
D = 2

X = np.random.randn(N, D)

# make first 50 points centrer around (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))

# make next 50 points center around (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# bias column(?)
Xb = np.concatenate((np.ones((N, 1)), X), axis=1)
Xc = np.concatenate((X, np.ones((N, 1))), axis=1)

# first half of targets category 0...
T = np.array([0] * 50 + [1] * 50)

# random weights (the extra column is the bias term)
w = np.random.randn(D + 1)

# Predictions Y is sigmoid of dot product
Y = expit(Xb.dot(w))


def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


print('Cross entropy error is:', calculate_cross_entropy(T, Y))

# use closed-form solution for logistic regression
# closed form will work as variances are same for each class in X
# therefore w depends only on means (of classes in X?)

# bias is 0 and each weight is four
# you would need to do the math
# to calculate this w manually
w = np.array([0, 4, 4])

Y = expit(Xb.dot(w))
print('Cross entropy error is:', calculate_cross_entropy(T, Y))
