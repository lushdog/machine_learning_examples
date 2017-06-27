import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

N = 4
D = 2

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
T = np.array([0, 1, 1, 0])

plt.title('XOR classes')
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

# as you can see from the viz, there is no
# line to split the two classes in T

# the way around this is to add another dimension
# so that it's a 3d problem, not a 2d problem,
# hyperplane can split the classes whereas a normal plane
# cannot
xy = np.matrix(X[:, 0] * X[:, 1]).T

# bias term
ones = np.ones((N, 1))

# concatenate X, xy and bias
# note since xy was created with np.matrix() we have
# to put wrap np.array() around np.concatenate() or else
# Y.shape is (1,4) instead of (4,)
Xb = np.array(np.concatenate((ones, xy, X), axis=1))


def classification_rate(T, P):
    return np.mean(T == P)


def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# random weights
w = np.random.randn(D + 2)

# predictions are sigmoid of dot-product
Y = expit(Xb.dot(w))

learning_rate = 0.01
errors = []
for t in range(5000):
    cost = calculate_cross_entropy(T, Y)
    if (t % 100 == 0):
        print(cost)
    errors.append(cost)

    w += learning_rate * ((Xb.T.dot(T - Y)) - (0.01 * w))
    Y = expit(Xb.dot(w))

plt.plot(errors)
plt.title('Cross-entroy per iteration')
plt.show()

print('Final w', w)
print('Classification rate', classification_rate(T, np.round(Y)))

