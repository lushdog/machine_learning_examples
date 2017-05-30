import numpy as np

N = 100
D = 2

# create some random data and bias column
X = np.random.randn(N, D)
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# create random weight vector
w = np.random.randn(D + 1)  # add one for bias column in Xb

z = Xb.dot(w)
print(z.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


print(sigmoid(z))
