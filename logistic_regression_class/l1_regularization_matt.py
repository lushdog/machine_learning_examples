import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# X will be a fat matrix
# we will use L1 regularization to find
N = 50
D = 50

# X is uninformly distributed around 0 from -5 to + 5
X = (np.random.random((N, D)) - 0.5) * 10

# only first 3 dimensions affect output, rest are 0 (don't affect output)
true_w = ([1, 0.5, 0.5] + [0] * (D - 3))

# generate Y
Y = np.round(expit(X.dot(true_w) + np.random.randn(N) * 0.5))

# random w
w = np.random.randn(D) / np.sqrt(D)

# no bias term as we've created the model above with true_w

# gradient descent with L1 regularization:
costs = []
learning_rate = 0.001
# l1 penalty (use different values to achieve best results)
# try 0 to see what happens with no regularization
l1 = 10

for t in range(5000):
    Y_hat = expit(X.dot(w))
    Y_delta = Y_hat - Y
    w = w - learning_rate * (X.T.dot(Y_delta) + l1 * np.sign(w))

    cost = -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)).mean() + l1 * np.abs(w).mean()
    costs.append(cost)

plt.plot(costs)
plt.show()

plt.plot(true_w, label='true_w')
plt.plot(w, label='w map')
plt.legend()
plt.show()
