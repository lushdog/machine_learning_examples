import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# lot of samples, not many dimensions
N = 1000
D = 2

# data is clustered around two radiuses
R_inner = 5
R_outer = 10

# half data depends on inner radius and
# other half on outer radius
R1 = np.random.randn(N // 2) + R_inner
# polar coordinates
theta = 2 * np.pi * np.random.random(N // 2)
# convert polar to x,y coordinates
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N // 2) + R_outer
theta = 2 * np.pi * np.random.random(N // 2)
# convert polar to x,y coordinates
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
# first half of targets are 0
# second half of targets are 1
T = np.array([0] * (N // 2) + [1] * (N // 2))

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()

# logistic regression may not be good for this
# as there is no line to seperate classes

# or is there?

# calculate radiuses
r = np.zeros((N, 1))
for i in range(N):
    r[i] = np.sqrt(X[i, :].dot(X[i, :]))

# concatenate radiuses and bias term with X
X = np.concatenate((np.ones((N, 1)), r, X), axis=1)
print(X)

# solve for w
w = np.random.random(D + 2)
Y = expit(X.dot(w))

'''
def calculate_cross_entropy(T, Y):
    return -np.mean(T * np.log(Y) + (1 - T) * np.log(1 - Y))
'''


def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# gradient descent with L2 regularization
learning_rate = 0.001
errors = []
for t in range(5000):
    errors.append(calculate_cross_entropy(T, Y))
    if (t % 100 == 0):
        print('Cost:', errors[len(errors) - 1])
    w += learning_rate * (X.T.dot(T - Y) - 0.01 * w)
    Y = expit(X.dot(w))

plt.plot(errors)
plt.title('Cross-Entropy')
plt.show()

print('Final w:', w)
print('Final classification rate:', 1 - np.abs(T - np.round(Y)).sum() / N)
