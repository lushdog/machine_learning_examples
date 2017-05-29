import numpy as np
from scipy.special import expit
from course_project_data_processing import get_data

X, Y = get_data()
D = X.shape[1]
W = np.random.randn(D)
# bias term in logistic is scalar and not a vector like in linear regression
b = 0


# use scipy.special.expit(a)
# def sigmoid(a):
    # return 1 / (1 + np.exp(-a))


def forward(X, w, b):
    '''
    sum(values time weights plus bias), then
    sigmoid (logistic function) clamps
    the sums to range from 0 to 1
    with y-intercept at 0.5
    '''
    print('X.dot(W) + b:', X.dot(W) + b)
    return expit(X.dot(W) + b)


def classification_score(Y, P):
    print('Y matches P:', Y == P)
    return np.mean(Y == P)


P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)
print('P_Y_given_X', P_Y_given_X)
print('Predictions:', predictions)
print('Y', Y)
print('Classification score:', classification_score(Y, predictions))
