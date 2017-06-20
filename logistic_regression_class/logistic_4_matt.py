import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# !!!!!!!!!!
# omly change to this file from logistic_3_matt.py
# is the addition of L2 regularization to calculation of w
# during gradient descent at bottom of this file

N = 100
D = 2

X = np.random.randn(N, D)

# create two gaussian clouds

# make first 50 points centrer around (-2, -2)
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))

# make next 50 points center around (2, 2)
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# bias column
Xb = np.concatenate((np.ones((N, 1)), X), axis=1)

# first half of targets category 0...
T = np.array([0] * 50 + [1] * 50)

# random weights (the extra column is the bias term)
w = np.random.randn(D + 1)
Y = expit(Xb.dot(w))


def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# Predictions Y_hat is sigmoid of dot product
learning_rate = 0.1
errors = []
for t in range(100):
    entropy = calculate_cross_entropy(T, Y)
    errors.append(entropy)
    if t % 10 == 0:
        print(entropy)

    # !!!!!
    # only change to this file is L2 regularization added to
    # calculation of w, you can compare values in w with 
    # and without L2reg and you will see lower values of w
    # when using L2 reg

    # w calculation without L2 regularization:
    # w += learning_rate * Xb.T.dot(T - Y)

    # w calcualtion with L2 regularization and
    # lambda set to 0.1:
    w += learning_rate * (Xb.T.dot(T - Y) - 0.1 * w)

    Y = expit(Xb.dot(w))

'''
plt.plot(errors)
plt.show()
'''

print('Final w:', w)

