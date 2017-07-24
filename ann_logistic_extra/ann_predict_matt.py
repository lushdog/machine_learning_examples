import numpy as np
import matplotlib.pyplot as plt
import process_matt as process

X, Y = process.get_data()

N = X.shape[0]
D = X.shape[1]
M = 5
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    pY = softmax(Z.dot(W2) + b2)
    assert np.allclose(pY.sum(axis=1), np.ones(len(X)))
    return pY


def classification_rate(Y, P):
    return np.mean(Y == P)


pY = forward(X, W1, b1, W2, b2)
P = np.argmax(pY, axis=1)
print(P)
assert(len(P) == len(Y))
print('classification_rate', classification_rate(Y, P))
