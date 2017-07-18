# build feed forward network without an API

import numpy as np
import matplotlib.pyplot as plt

Nclass = 500  # num observations of each class
D = 2         # num dimensions
# three gaussian clouds each w/ different centers
X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])

Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
X = np.vstack((X1, X2, X3))

plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
plt.show()

D = 2  # num dimensions (dupe)
M = 3  # 1 hidden layer with size of 3
K = 3  # num classes

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def forward(X, W1, b1, W2, b2):
    # sigmoid from inputs to hidden layer -
    # Z are the values at the hidden layer
    # Z shape is (N, M)
    Z = sigmoid(X.dot(W1) + b1)

    # softmax for hidden layer to outputs -
    # Each column  of Y is the probability of the row being a
    # sample of each class, thus Y.shape is (N, K)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    pY = expA / expA.sum(axis=1, keepdims=True)
    assert np.allclose(pY.sum(axis=1), np.ones(len(X)))
    return pY


def classification_rate(Y, P):
    n_correct = 0
    n_total = 1
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


pY = forward(X, W1, b1, W2, b2)
P = np.argmax(pY, axis=1)
assert(len(P) == len(Y))
print("Classification rate of random weights/biases:",
      classification_rate(Y, P))
